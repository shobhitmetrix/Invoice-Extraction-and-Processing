

import requests
import json
import base64
import re
import os
import logging
from datetime import datetime
import glob
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
import io
import sys
import argparse


# =========================================================
# LOGGING
# =========================================================

def setup_logging(input_folder):
    log_folder = os.path.join(input_folder, "invoice_processing_logs")

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        log_folder, f"invoice_processing_{timestamp}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


# =========================================================
# IMAGE HELPERS
# =========================================================

def auto_rotate_image(image_path):
    try:
        img = Image.open(image_path)

        if hasattr(img, "_getexif"):
            exif = img._getexif()
            if exif:
                orientation = exif.get(274)
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)

        img_bytes = io.BytesIO()

        if image_path.lower().endswith(".png"):
            img.save(img_bytes, format="PNG")
        else:
            img.save(img_bytes, format="JPEG", quality=95)

        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        logging.warning(f"Could not rotate image {image_path}: {e}")
        return None


def encode_image_to_base64(image_path):
    try:
        rotated = auto_rotate_image(image_path)
        if rotated:
            return base64.b64encode(rotated.read()).decode("utf-8")
        else:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Encoding failed: {e}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# =========================================================
# LINE ITEM NORMALIZATION
# =========================================================

def normalize_line_items(extracted_data):
    if not isinstance(extracted_data, dict):
        return extracted_data

    if "line_items" not in extracted_data:
        return extracted_data

    merged = []
    buffer = None

    for item in extracted_data["line_items"]:
        if item.get("product_description") or item.get("commodity"):
            if buffer:
                merged.append(buffer)
            buffer = item.copy()
            buffer.pop("commodity", None)
        else:
            if buffer:
                for k, v in item.items():
                    if v and not buffer.get(k):
                        buffer[k] = v
            else:
                merged.append(item)

    if buffer:
        merged.append(buffer)

    extracted_data["line_items"] = merged
    return extracted_data


# =========================================================
# ULTRATECH FIXES
# =========================================================

def fix_ultratech_fields(extracted_data):
    if not isinstance(extracted_data, dict):
        return extracted_data

    vendor = extracted_data.get("invoice_details", {}).get("vendor_name", "")
    if not any(x in vendor.lower() for x in ["ultratech", "birla"]):
        return extracted_data

    items = extracted_data.get("line_items", [])
    if not isinstance(items, list):
        return extracted_data

    rates = []

    for item in items:
        for field in ["basic_mt_l_kg", "basic_amount"]:
            try:
                val = float(str(item.get(field, "")).replace(",", ""))
                rates.append((field, val))
            except:
                pass

    plausible = [v for f, v in rates if f == "basic_mt_l_kg" and v > 1000]
    if plausible:
        best_rate = max(set(plausible), key=plausible.count)
    else:
        return extracted_data

    for item in items:
        try:
            qty = float(item.get("quantity", "1").replace(",", ""))
        except:
            qty = 1

        item["basic_mt_l_kg"] = f"{best_rate:,.2f}"
        item["basic_amount"] = f"{qty * best_rate:,.2f}"

        item.pop("cash_discount", None)
        item.pop("trade_discount", None)

    return extracted_data


# =========================================================
# JSON CLEANING
# =========================================================

def clean_json_string(text):
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def extract_json_from_text(text):
    """Extract JSON from text that may be wrapped in markdown or have extra characters."""
    if not text:
        return None

    # Remove markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try to parse directly
    try:
        return json.loads(text)
    except:
        pass

    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass

    return None


def clean_json_output(raw_output):
    """
    Extract the actual invoice JSON from the API response.
    Handles both the direct dict and the nested 'choices' format.
    """
    if not raw_output:
        return None

    # If it's already a dict and looks like the invoice data (has invoice_details, etc.), return it
    if isinstance(raw_output, dict):
        if "invoice_details" in raw_output or "line_items" in raw_output:
            return raw_output

        # Standard API response format
        if "choices" in raw_output and raw_output["choices"]:
            content = raw_output["choices"][0].get("message", {}).get("content", "")
            if content:
                parsed = extract_json_from_text(content)
                if parsed:
                    return parsed

        # If there is a top-level "content" field
        if "content" in raw_output and isinstance(raw_output["content"], str):
            parsed = extract_json_from_text(raw_output["content"])
            if parsed:
                return parsed

        # Fallback: return raw dict (maybe it's already correct)
        return raw_output

    # If it's a string, try to extract JSON
    if isinstance(raw_output, str):
        return extract_json_from_text(raw_output)

    return None

# =========================================================
# FILE HELPERS
# =========================================================

def save_json_output(data, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved: {path}")


def get_image_files(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def get_pending_images(images, output_folder):
    pending = []
    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        if not os.path.exists(os.path.join(output_folder, f"{name}.json")):
            pending.append(img)
    return pending


def extract_invoice_data_api(image_path, api_key, use_structured_prompt=False, vendor_name=None):
    """
    Extract invoice data using Qubrid Multimodal Chat API.
    If vendor_name contains 'UltraTech' or 'Birla', a specialised prompt that handles
    the three‑row‑per‑product layout is used.
    """
    url = "https://platform.qubrid.com/api/v1/qubridai/multimodal/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    base64_image = encode_image_to_base64(image_path)

    # ----- UltraTech / Birla White specific prompt -----
    if vendor_name and any(x in vendor_name.lower() for x in ['ultratech', 'birla', 'birla white']):
        prompt = """
        This invoice is from **UltraTech Cement (Birla White)**. Its table has a **unique three‑row layout**:

        ```
        Row A: Commodity | Product | HSN Code
        Row B: Qty In MT/L/KG | No of Bag/Pouch | Basic MT/L/KG
        Row C: CashDisc MT/L/KG | TradeDis MT/L/KG | Basic Amt
        ```

        **For one product, these three rows appear stacked vertically.**  
        You **MUST combine them into a SINGLE line item** with the following fields:

        - commodity (from Row A, first column)
        - product_description (from Row A, second column)
        - hsn_code (from Row A, third column)
        - quantity (from Row B, first column)
        - bags_pouches (from Row B, second column)
        - basic_mt_l_kg (from Row B, third column)   ← the extended amount (Qty × Unit Price)
        - cash_discount (from Row C, first column)
        - trade_discount (from Row C, second column)
        - basic_amount (from Row C, third column)    ← final amount after discounts (often blank or dashes)

        - If a cell contains dashes ("--") or is empty, use an empty string "".
        - Do **NOT** split a product into multiple JSON objects.
        - The invoice may have more than one product; list each as a separate object.

        Also extract:
        - Invoice number, date, vendor/customer details
        - Totals (subtotal, tax, total amount)

        Return **ONLY** a valid JSON object with this exact structure:

        {
          "invoice_details": {
            "invoice_number": "...",
            "date": "...",
            "vendor_name": "UltraTech Cement Limited",
            "vendor_address": "...",
            "customer_name": "...",
            "customer_address": "..."
          },
          "line_items": [
            {
              "commodity": "...",
              "product_description": "...",
              "hsn_code": "...",
              "quantity": "...",
              "bags_pouches": "...",
              "basic_mt_l_kg": "...",
              "cash_discount": "...",
              "trade_discount": "...",
              "basic_amount": "..."
            }
          ],
          "totals": {
            "subtotal": "...",
            "tax": "...",
            "total": "..."
          }
        }

        Do not add any extra text, markdown, or explanations.
        """
    else:
        # ----- Default generic prompt (unchanged) -----
        if use_structured_prompt:
            prompt = """
            Extract all invoice data from the provided image and return ONLY a valid JSON object. 
            Structure should be:
            {
                "invoice_details": {
                    "invoice_number": "",
                    "date": "",
                    "due_date": "",
                    "vendor_name": "",
                    "vendor_address": "",
                    "customer_name": "",
                    "customer_address": ""
                },
                "line_items": [
                    {
                        "description": "",
                        "quantity": "",
                        "unit_price": "",
                        "amount": ""
                    }
                ],
                "totals": {
                    "subtotal": "",
                    "tax": "",
                    "total": ""
                }
            }
            Return ONLY the JSON, no additional text or explanations.
            """
        else:
            prompt = "Extract all invoice data from this invoice image and return it as a valid JSON object with the following structure: { 'invoice_details': { ... }, 'line_items': [ ... ], 'totals': { ... } }. Make sure the output is pure JSON that can be parsed by json.loads(). Do not include any additional text."

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    data = {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 16384,
        "seed": 42,
        "top_k": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
        "stream": False
    }

    try:
        logging.info(f"Sending API request for {os.path.basename(image_path)} (vendor: {vendor_name})...")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code != 200:
            logging.error(f"API returned error: {response.status_code}")
            return None
        try:
            result = response.json()
            return result
        except json.JSONDecodeError:
            return process_streaming_response(response.text)
    except Exception as e:
        logging.error(f"API request failed: {e}")
        return None


def process_streaming_response(stream_text):
    """Process Server-Sent Events (SSE) streaming response"""
    logging.info("Processing streaming response...")
    
    full_content = ""
    lines = stream_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_content = line[6:]  # Remove 'data: ' prefix
            
            # Skip empty lines or [DONE] message
            if not data_content or data_content == '[DONE]':
                continue
                
            try:
                chunk_data = json.loads(data_content)
                
                # Extract content from the chunk
                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                    delta = chunk_data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        full_content += delta['content']
                        
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse chunk: {e}")
                continue
    
    logging.info(f"Extracted content from stream: {full_content[:200]}...")
    
    # Return the assembled content in the expected format
    if full_content:
        return {
            "choices": [
                {
                    "message": {
                        "content": full_content
                    }
                }
            ]
        }
    else:
        logging.error("No content extracted from streaming response")
        return None


# =========================================================
# PROCESSING
# =========================================================

def process_single_image(image_path, api_key, output_folder):
    image_name = os.path.basename(image_path)
    logging.info(f"Processing {image_name}")

    result = extract_invoice_data_api(image_path, api_key)
    cleaned = clean_json_output(result)

    if isinstance(cleaned, dict):
        base = os.path.splitext(image_name)[0]
        save_json_output(cleaned, output_folder, f"{base}.json")
        return {"status": "success", "image": image_name}

    return {"status": "failed", "image": image_name}


def process_images_parallel(images, api_key, output_folder, max_workers=2):
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_image, img, api_key, output_folder
            )
            for img in images
        ]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--api_key", required=True)
    args = parser.parse_args()

    setup_logging(args.input_folder)

    images = get_image_files(args.input_folder)
    pending = get_pending_images(images, args.output_folder)

    if not pending:
        logging.info("Nothing to process")
        return

    results = process_images_parallel(
        pending, args.api_key, args.output_folder
    )

    logging.info("Processing complete")


if __name__ == "__main__":
    main()