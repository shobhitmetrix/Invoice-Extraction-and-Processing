


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
import requests
import sys
import argparse


def setup_logging(input_folder):
   
    
    # Create log folder inside the input folder
    log_folder = os.path.join(input_folder, "invoice_processing_logs")
    
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"invoice_processing_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    
    return log_file  

def auto_rotate_image(image_path):
    """Automatically rotate image based on EXIF orientation tag"""
    try:
        # Open the image with PIL
        img = Image.open(image_path)
        
        # Check for EXIF orientation tag
        if hasattr(img, '_getexif'):
            exif = img._getexif()
            if exif is not None:
                orientation = exif.get(274)  # 274 is the EXIF tag for orientation
                
                # Rotate image based on orientation tag
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
        
        # Convert back to bytes
        img_bytes = io.BytesIO()
        
        # Save in JPEG format (preserve original format if needed)
        if image_path.lower().endswith('.png'):
            img.save(img_bytes, format='PNG')
        else:
            # Default to JPEG for other formats
            img.save(img_bytes, format='JPEG', quality=95)
        
        img_bytes.seek(0)
        return img_bytes
        
    except Exception as e:
        logging.warning(f"Could not rotate image {image_path}: {e}")
        # Return None if rotation fails, will use original image
        return None

def extract_company_name_from_logo(image_path, api_key, width_ratio=0.35, height_ratio=0.22):
    """
    Crop the top-left region (where the logo resides) and use Qubrid multimodal API
    to read ONLY the company/brand name from the logo.
    
    Adds robustness by using EXIF-aware transpose. Returns a clean company string or None.
    """
    try:
        # --- Crop top-left region where logos usually are ---
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)  # handle EXIF orientation safely
        w, h = img.size
        crop_box = (0, 0, int(w * width_ratio), int(h * height_ratio))
        logo_crop = img.crop(crop_box)

        # Encode cropped logo as base64 (JPEG)
        buf = io.BytesIO()
        logo_crop.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        logo_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # --- Build Qubrid prompt ---
        url = "https://platform.qubrid.com/api/v1/qubridai/multimodal/chat"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        prompt = (
            'Identify the brand/company name visible in this logo/cropped header. '
            'Return ONLY valid JSON in the form: {"company_from_logo": "<name or null>"} '
            "Use proper casing (e.g., 'Asian Paints'). If uncertain, return null."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{logo_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        data = {
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256,
            "top_k": 1,
            "stream": False,
        }

        logging.info("Sending logo-only request to extract company name...")
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code != 200:
            logging.warning(
                f"Logo API call failed: {resp.status_code} | {resp.text[:300]}"
            )
            return None

        # Reuse your existing JSON cleaning utilities for consistency
        raw = resp.json()
        parsed = clean_json_output(raw)
        if isinstance(parsed, dict):
            # Handle when the model puts the JSON in the usual choices->message->content
            if "company_from_logo" in parsed:
                val = parsed["company_from_logo"]
                if isinstance(val, str):
                    return val.strip() or None
                return None
            # Or when the content is a JSON string
            content = None
            if "choices" in parsed and parsed["choices"]:
                content = parsed["choices"][0].get("message", {}).get("content")
            if isinstance(content, str):
                content_parsed = extract_json_from_text(content)
                if isinstance(content_parsed, dict) and "company_from_logo" in content_parsed:
                    val = content_parsed["company_from_logo"]
                    if isinstance(val, str):
                        return val.strip() or None
                    return None

        # Fallback: try to extract directly from text if any
        if isinstance(parsed, str):
            maybe = extract_json_from_text(parsed)
            if isinstance(maybe, dict) and "company_from_logo" in maybe:
                val = maybe["company_from_logo"]
                if isinstance(val, str):
                    return val.strip() or None
                return None

        return None
    except Exception as e:
        logging.warning(f"Logo extraction error for {os.path.basename(image_path)}: {e}")
        return None

def normalize_asian_paints_line_items(items):
    """
    Normalize Asian Paints line items WITHOUT computing values.
    - Always clears discount fields: "in_bill_disc", "in_bill_disc_2", "cash_disc".
    - Ignores 'tax_amount' entirely to avoid misalignment due to spacing issues:
        * Forces row["tax_amount"] = ""
        * Right-aligns ONLY the remaining two amount columns:
            ["taxable_amount", "total_amount"]
      so the rightmost valid numeric becomes total_amount, the preceding one becomes taxable_amount.
    - Robust numeric sanitization: removes commas and ALL whitespace (incl. non-breaking spaces).
    - No inference or calculations.
    """
    if not isinstance(items, list):
        return items

    import re

    KEYS = [
        "material_hsn",
        "description",
        "qty",
        "packs",
        "volume_kg_ltr_m",
        "rate_inr_per_unit",
        "value",
        "in_bill_disc",
        "in_bill_disc_2",
        "cash_disc",
        "taxable_amount",
        "tax_amount",
        "total_amount",
    ]

    NUMERIC_LIKE = {
        "qty",
        "volume_kg_ltr_m",
        "rate_inr_per_unit",
        "value",
        "in_bill_disc",
        "in_bill_disc_2",
        "cash_disc",
        "taxable_amount",
        "tax_amount",
        "total_amount",
    }

    # Strict numeric: optional leading '-', digits, optional one decimal part (NO trailing '-')
    STRICT_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

    # Matches ANY Unicode whitespace (incl. non-breaking spaces) so we can strip them from numerics
    ANY_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

    def _clean_str(x):
        return "" if x is None else str(x)

    def _sanitize_numeric_like(s):
        """
        Remove thousands separators and ALL whitespace (incl. non-breaking \u00A0) from numeric-like text.
        Keep leading '-' if present; trailing '-' (discount-like) will fail numeric validation and be ignored.
        """
        if s is None:
            return ""
        s = str(s)
        s = s.replace(",", "")
        s = ANY_WS_RE.sub("", s)
        return s.strip()

    def _is_clean_number(s):
        if s is None:
            return False
        s = str(s).strip()
        return bool(STRICT_NUMERIC_RE.match(s))

    normalized = []
    for raw in items:
        row = {k: "" for k in KEYS}
        if isinstance(raw, dict):
            # Copy & sanitize
            for k in KEYS:
                if k in raw:
                    v = _clean_str(raw[k])
                    if k in NUMERIC_LIKE and v != "":
                        v = _sanitize_numeric_like(v)
                    else:
                        v = v.strip()
                    row[k] = v

            # 1) Always clear discount fields (ignore them)
            row["in_bill_disc"] = ""
            row["in_bill_disc_2"] = ""
            row["cash_disc"] = ""

            # 2) Ignore tax_amount completely to avoid misalignment
            #    We'll right-align only between taxable_amount and total_amount.
            row["tax_amount"] = ""

            # Collect valid numerics from taxable_amount and total_amount ONLY
            captured = []
            for k in ["taxable_amount", "total_amount"]:
                v = row.get(k, "")
                if not v:
                    continue
                v_san = _sanitize_numeric_like(v)
                if _is_clean_number(v_san):
                    captured.append(v_san)

            # Right-align into [taxable_amount, total_amount]
            # Examples:
            #   ["14612.19"] -> taxable="", total="14612.19"
            #   ["14612.19","2630.19"] -> taxable="14612.19", total="2630.19"
            new_taxable = captured[-2] if len(captured) >= 2 else ""
            new_total = captured[-1] if len(captured) >= 1 else ""
            row["taxable_amount"] = new_taxable
            row["total_amount"] = new_total

        normalized.append(row)

    return normalized


def encode_image_to_base64(image_path):
    """Encode image to base64 string with auto-rotation"""
    try:
        # Try to auto-rotate the image first
        rotated_img_bytes = auto_rotate_image(image_path)
        
        if rotated_img_bytes:
            # Use rotated image
            return base64.b64encode(rotated_img_bytes.read()).decode('utf-8')
        else:
            # Fallback to original image
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
                
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        # Fallback to original image
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def extract_invoice_data_api(image_path, api_key, use_structured_prompt=False, company_from_logo=None):
    """Extract invoice data using Qubrid Multimodal Chat API with streaming handling.
    If company_from_logo indicates Asian Paints, enforce a strict line-items schema and
    non-shifting rules in the prompt, and REQUIRE extraction of tax columns when visible.
    """

    # API endpoint and headers
    url = "https://platform.qubrid.com/api/v1/qubridai/multimodal/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)

    # --- Asian Paints line-items schema (enforced when logo indicates Asian Paints) ---
    asian_paints_hint = ""
    if company_from_logo and str(company_from_logo).strip().lower().replace(" ", "") in {"asianpaints", "asianpaint"}:
        asian_paints_hint = """
        IMPORTANT (Asian Paints): Column headers may be missing, cropped, or unreadable.
        Regardless of visible headers, you MUST structure every element in "line_items" with EXACTLY these keys and order:
        "line_items": [
          {
            "material_hsn": "",
            "description": "",
            "qty": "",
            "packs": "",
            "volume_kg_ltr_m": "",
            "rate_inr_per_unit": "",
            "value": "",
            "in_bill_disc": "",
            "in_bill_disc_2": "",
            "cash_disc": "",
            "taxable_amount": "",
            "tax_amount": "",
            "total_amount": ""
          }
        ]

        CRITICAL RULES:
        - NEVER shift values left or right when any field is blank. If a column value is missing for a row, set that key to "".
        - Keep ALL keys for EVERY item, in the SAME order; do not omit any key.
        - Do NOT compute, infer, or estimate any value (e.g., tax_amount or total_amount). If a value is not readable in the row, return "".
        - You MUST read "tax_amount" and "total_amount" from the SAME ROW’s rightmost two numeric cells (not from the Summary box).
        - If the values are visible in the row, DO NOT leave them blank. Read the exact numerals (preserve minus sign if present).
        - Do NOT copy "Taxable Amount", "Tax Amount" or "Total Amount" from the Summary grid; only per-line row cells.

        Formatting:
        - For numeric-like fields (qty, volume_kg_ltr_m, rate_inr_per_unit, value, in_bill_disc, in_bill_disc_2, cash_disc, taxable_amount, tax_amount, total_amount),
          return plain text numbers with optional decimal point and optional trailing minus for discounts (no currency symbols). Example: "2", "2607.00", "651.75-", "14612.19".
        - If a value is split across line breaks, join without spaces.
        """

    # Choose prompt based on preference
    if use_structured_prompt:
        prompt = f"""
        Extract all invoice data from the provided image and return ONLY a valid JSON object. 
        Structure should be:
        {{
            "invoice_details": {{
                "invoice_number": "",
                "date": "",
                "due_date": "",
                "vendor_name": "",
                "vendor_address": "",
                "customer_name": "",
                "customer_address": ""
            }},
            "line_items": [
                {{
                    "description": "",
                    "quantity": "",
                    "unit_price": "",
                    "amount": ""
                }}
            ],
            "totals": {{
                "subtotal": "",
                "tax": "",
                "total": ""
            }}
        }}
        {asian_paints_hint}
        Return ONLY the JSON, no additional text or explanations.
        """
    else:
        prompt = (
            "Extract all invoice data from this invoice image and return it as a valid JSON object with the following structure: "
            "{ 'invoice_details': { ... }, 'line_items': [ ... ], 'totals': { ... } }. "
            "Make sure the output is pure JSON that can be parsed by json.loads(). Do not include any additional text."
            f"{asian_paints_hint}"
        )

    # Prepare the messages in chat format
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

    # Prepare the payload for multimodal chat
    data = {
        # "model": "Qwen/Qwen3-VL-8B-Instruct",
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 16384,
        "seed": 42,
        "top_k": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
        "stream": False
    }

    try:
        logging.info(f"Sending API request for {os.path.basename(image_path)}...")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        logging.info(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            logging.error(f"API returned error status: {response.status_code}")
            logging.error(f"Response headers: {dict(response.headers)}")
            logging.error(f"Response text: {response.text[:500]}...")
            return None

        try:
            result = response.json()
            logging.info("✅ Successfully parsed JSON response")
            return result
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            logging.info("Response appears to be streaming format, processing as SSE...")
            return process_streaming_response(response.text)

    except requests.exceptions.Timeout:
        logging.error("API request timed out after 60 seconds")
        return None
    except requests.exceptions.ConnectionError:
        logging.error("Connection error - check your internet connection")
        return None
    except requests.exceptions.RequestException as e:
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

def clean_json_string(json_str):
    """Clean JSON string by removing common issues"""
    if not json_str:
        return json_str
    
    # Remove control characters
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing commas between objects in arrays
    json_str = re.sub(r'}\s*{', '},{', json_str)
    
    # Remove extra whitespace
    json_str = re.sub(r'\s+', ' ', json_str)
    
    return json_str.strip()

def fix_json_string(json_str):
    """Attempt to fix common JSON formatting issues"""
    if not json_str or not isinstance(json_str, str):
        return json_str
    
    try:
        # First check if it's already valid
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        logging.debug(f"JSON needs fixing: {e}")
    
    fixed = json_str
    
    # Fix 1: Replace single quotes with double quotes (carefully)
    # Only replace quotes that aren't already escaped and are around keys or string values
    lines = fixed.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Use regex to replace unescaped single quotes that look like they're for JSON keys/values
        # This pattern looks for : followed by optional space then single quote
        line = re.sub(r':\s*\'', ': "', line)
        # This pattern looks for single quote followed by optional space then :
        line = re.sub(r'\'\s*:', '":', line)
        # This pattern looks for single quote at end of value (before comma or end of object)
        line = re.sub(r'\'(?=\s*[,}])', '"', line)
        # This pattern looks for single quote at beginning of value
        line = re.sub(r'(?<=:\s*)\'', '"', line)
        fixed_lines.append(line)
    
    fixed = '\n'.join(fixed_lines)
    
    # Fix 2: Remove trailing commas
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    
    # Fix 3: Fix True/False/Null
    fixed = re.sub(r':\s*True\b', ': true', fixed)
    fixed = re.sub(r':\s*False\b', ': false', fixed)
    fixed = re.sub(r':\s*Null\b', ': null', fixed)
    fixed = re.sub(r':\s*NULL\b', ': null', fixed)
    
    # Fix 4: Remove comments
    fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)
    
    # Fix 5: Remove extra commas between array items
    fixed = re.sub(r',\s*,', ',', fixed)
    
    return fixed

def extract_json_from_text(text):
    """Extract JSON from text response with improved parsing"""
    if not text:
        return None
    
    logging.debug("Attempting to extract JSON from text...")
    
    # First, clean the text - remove markdown code blocks and extra whitespace
    clean_text = text.strip()
    
    # Remove markdown code blocks
    if clean_text.startswith('```json'):
        clean_text = clean_text[7:]
    elif clean_text.startswith('```'):
        clean_text = clean_text[3:]
    if clean_text.endswith('```'):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()
    
    # Try to parse the entire cleaned text as JSON first
    try:
        parsed = json.loads(clean_text)
        logging.info("✅ Successfully parsed entire cleaned text as JSON!")
        return parsed
    except json.JSONDecodeError:
        logging.debug("Entire cleaned text is not valid JSON, trying extraction...")
    
    # The text might be a JSON string that starts with unexpected characters
    # Try to find the first { or [ and parse from there
    for start_char in ['{', '[']:
        start_idx = clean_text.find(start_char)
        if start_idx != -1:
            try:
                # Try to parse from the start character to the end
                json_str = clean_text[start_idx:]
                parsed = json.loads(json_str)
                logging.info("✅ Successfully parsed JSON from start character!")
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Try to find the last } or ] and work backwards
    for end_char in ['}', ']']:
        end_idx = clean_text.rfind(end_char)
        if end_idx != -1:
            try:
                # Find matching opening bracket
                open_char = '{' if end_char == '}' else '['
                # Try to find the matching opening bracket
                # This is a simplified approach - we'll take a substring
                start_idx = clean_text.rfind(open_char, 0, end_idx)
                if start_idx != -1:
                    json_str = clean_text[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    logging.info("✅ Successfully parsed JSON by finding matching brackets!")
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue
    
    # If we still can't parse it, try to find any JSON-like structure
    # Look for patterns that look like JSON objects
    json_candidates = []
    
    # Find all potential JSON objects
    pattern = r'\{[^{}]*?(?:\{[^{}]*\}[^{}]*?)*\}'
    matches = re.findall(pattern, clean_text, re.DOTALL)
    
    for match in matches:
        if len(match) > 20:  # Reasonable minimum length for JSON
            json_candidates.append(match)
    
    # Sort by length (longest first as they're more likely to be complete)
    json_candidates.sort(key=len, reverse=True)
    
    for candidate in json_candidates:
        try:
            # Clean the candidate
            cleaned = clean_json_string(candidate)
            parsed = json.loads(cleaned)
            logging.info("✅ Successfully parsed JSON candidate!")
            return parsed
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed = fix_json_string(candidate)
            if fixed:
                try:
                    parsed = json.loads(fixed)
                    logging.info("✅ Successfully parsed fixed JSON candidate!")
                    return parsed
                except json.JSONDecodeError:
                    continue
    
    # Last resort: check if the text itself is a string representation we can convert
    # Sometimes the API returns a string that's already a Python dict representation
    if clean_text.startswith("{") and clean_text.endswith("}"):
        # Try to replace single quotes with double quotes (carefully)
        try:
            # First try with eval (be cautious)
            import ast
            parsed = ast.literal_eval(clean_text)
            logging.info("✅ Successfully parsed using ast.literal_eval!")
            return parsed
        except (SyntaxError, ValueError):
            pass
    
    logging.warning(f"Response is not valid JSON. Text preview: {clean_text[:200]}")
    return text


def clean_json_output(raw_output):
    """Clean and extract JSON from the API response"""
    if not raw_output:
        return None
    
    logging.debug(f"Raw output type: {type(raw_output)}")
    
    # If the response is the full API response structure
    if isinstance(raw_output, dict):
        logging.debug("Processing dictionary response...")
        
        # Check if this is already the structured data we want
        # Look for keys that indicate it's already the extracted data
        expected_keys = ['invoice_details', 'line_items', 'totals']
        if any(key in raw_output for key in expected_keys):
            logging.info("✅ Response already has expected structure, returning as is")
            return raw_output
        
        # Extract content from various possible response formats
        content = None
        
        # Format 1: Standard choices format
        if 'choices' in raw_output and len(raw_output['choices']) > 0:
            if isinstance(raw_output['choices'][0], dict):
                if 'message' in raw_output['choices'][0]:
                    content = raw_output['choices'][0]['message'].get('content', '')
                elif 'text' in raw_output['choices'][0]:
                    content = raw_output['choices'][0]['text']
                else:
                    # Try to get any string value from the choice
                    for key, value in raw_output['choices'][0].items():
                        if isinstance(value, str) and len(value) > 10:
                            content = value
                            break
        
        # Format 2: Direct content or text keys
        if not content:
            for key in ['content', 'text', 'result', 'output', 'response']:
                if key in raw_output:
                    content = raw_output[key]
                    break
        
        # Format 3: The entire response might be the content
        if not content:
            # Check if any top-level string values look like JSON
            for key, value in raw_output.items():
                if isinstance(value, str) and len(value) > 50 and ('{' in value or '[' in value):
                    content = value
                    break
        
        if content:
            logging.debug(f"Extracted content: {str(content)[:200]}...")
            
            # If content is already a dict (some APIs return nested dicts)
            if isinstance(content, dict):
                return content
            
            # If content is a string, try to extract JSON
            if isinstance(content, str):
                result = extract_json_from_text(content)
                # If extract_json_from_text returned the original string, try one more time with aggressive cleaning
                if isinstance(result, str) and result == content:
                    logging.debug("Initial extraction failed, trying aggressive cleaning...")
                    # Remove any non-JSON prefixes/suffixes
                    lines = content.split('\n')
                    cleaned_lines = []
                    in_json = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('{') or stripped.startswith('['):
                            in_json = True
                        if in_json:
                            cleaned_lines.append(line)
                        if stripped.endswith('}') or stripped.endswith(']'):
                            in_json = False
                    
                    if cleaned_lines:
                        cleaned_content = '\n'.join(cleaned_lines)
                        result = extract_json_from_text(cleaned_content)
                
                return result
        
        # If we couldn't extract content but raw_output has data, return it as is
        logging.warning("Could not extract content, returning raw output")
        return raw_output
    
    # If it's a string, extract JSON from it
    elif isinstance(raw_output, str):
        logging.debug(f"Processing string response: {raw_output[:200]}...")
        return extract_json_from_text(raw_output)
    
    logging.error(f"Unexpected response type: {type(raw_output)}")
    return raw_output

def save_json_output(data, output_folder, filename):
    """Save JSON data to file in specified output folder"""
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Results saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        return False



def display_extracted_data(extracted_data, image_name):
    """Display the extracted invoice data in a readable format"""
    if not isinstance(extracted_data, dict):
        logging.warning(f"No valid data to display for {image_name}")
        return
    
    print("\n" + "="*50)
    print(f"EXTRACTED INVOICE DATA - {image_name}")
    print("="*50)
    
    # Invoice Details
    if 'invoice_details' in extracted_data:
        print("\n📄 INVOICE DETAILS:")
        details = extracted_data['invoice_details']
        
        # Display company name from logo first if available
        if 'company_name_from_logo' in details and details['company_name_from_logo']:
            print(f"  Company Name (from logo): {details['company_name_from_logo']}")
        
        for key, value in details.items():
            # Skip company_name_from_logo as we already displayed it
            if key != 'company_name_from_logo' and value:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Line Items
    if 'line_items' in extracted_data and extracted_data['line_items']:
        print("\n📦 LINE ITEMS:")
        for i, item in enumerate(extracted_data['line_items'], 1):
            print(f"  Item {i}:")
            for key, value in item.items():
                if value:
                    print(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Totals
    if 'totals' in extracted_data:
        print("\n💰 TOTALS:")
        totals = extracted_data['totals']
        for key, value in totals.items():
            if value:
                print(f"  {key.title()}: {value}")
    
    print("="*50)


def test_api_connection(api_key):
    """Test basic API connectivity"""
    url = "https://platform.qubrid.com/api/v1/qubridai/multimodal/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello, are you working?"}],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False  # Disable streaming for test
    }
    
    try:
        response = requests.post(url, headers=headers, json=test_data, timeout=10)
        logging.info(f"Test API Status: {response.status_code}")
        if response.status_code == 200:
            logging.info("✅ API connection successful!")
            return True
        else:
            logging.error(f"❌ API test failed with status: {response.status_code}")
            logging.error(f"Response: {response.text[:500]}")
            return False
    except Exception as e:
        logging.error(f"❌ API test failed: {e}")
        return False

def get_image_files(folder_path):
    """Get all image files from the folder without duplicates"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    image_files = []
    
    for extension in image_extensions:
        # Use case-insensitive glob pattern to avoid duplicates
        pattern = os.path.join(folder_path, f"*{extension}")
        files = glob.glob(pattern, recursive=False)
        image_files.extend(files)
    
    # Remove duplicates by converting to set and back to list
    unique_files = list(set(image_files))
    
    # Sort for consistent processing order
    return sorted(unique_files)

def get_pending_images(image_files, output_folder):
    """Filter images that don't have corresponding JSON files yet"""
    pending_images = []
    
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        json_path = os.path.join(output_folder, f"{base_name}.json")
        
        if not os.path.exists(json_path):
            pending_images.append(image_path)
        else:
            logging.info(f"⏭️  Skipping {image_name} - JSON already exists")
    
    return pending_images


def process_single_image(image_path, api_key, output_folder, use_structured_prompt=False):
    """Process a single image and return results.
    - Detect company from logo
    - Extract invoice JSON (pass company to API for schema enforcement)
    - Normalize Asian Paints line items to avoid column shifting on blanks
    - Save & display results
    """


    image_name = os.path.basename(image_path)
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing: {image_name}")
    logging.info(f"{'='*60}")

    try:
        # First, extract company name from logo
        company_name_from_logo = extract_company_name_from_logo(image_path, api_key)

        # Extract invoice data (now passing company_from_logo to enforce schemas when needed)
        result = extract_invoice_data_api(
            image_path,
            api_key,
            use_structured_prompt,
            company_from_logo=company_name_from_logo
        )

        if result:
            # Clean and extract JSON from response
            cleaned_result = clean_json_output(result)

            if cleaned_result and isinstance(cleaned_result, dict):
                logging.info(f"✅ Successfully extracted JSON data from {image_name}!")

                # Add company name from logo to the invoice details
                if company_name_from_logo:
                    if 'invoice_details' not in cleaned_result:
                        cleaned_result['invoice_details'] = {}
                    cleaned_result['invoice_details']['company_name_from_logo'] = company_name_from_logo
                    logging.info(f"✅ Added company name from logo: {company_name_from_logo}")

                # --- Normalize Asian Paints line_items to prevent blank-column shifting ---
                if company_name_from_logo and str(company_name_from_logo).strip().lower().replace(" ", "") in {"asianpaints", "asianpaint"}:
                    if isinstance(cleaned_result.get("line_items"), list):
                        cleaned_result["line_items"] = normalize_asian_paints_line_items(cleaned_result["line_items"])

                # Generate output filename
                base_name = os.path.splitext(image_name)[0]
                json_filename = f"{base_name}.json"

                # Save JSON file
                save_json_output(cleaned_result, output_folder, json_filename)

                # Display results (assumes display_extracted_data exists in your code)
                try:
                    display_extracted_data(cleaned_result, image_name)
                except Exception:
                    # If display_extracted_data isn't present in the current file, skip gracefully
                    pass

                return {
                    "status": "success",
                    "image_name": image_name,
                    "json_filename": json_filename,
                    "data": cleaned_result
                }
            else:
                logging.warning(f"❌ Failed to extract valid JSON from {image_name}")
                # Try with structured prompt as fallback
                if not use_structured_prompt:
                    logging.info(f"Trying structured prompt for {image_name}...")
                    return process_single_image(image_path, api_key, output_folder, use_structured_prompt=True)
                else:
                    return {
                        "status": "failed",
                        "image_name": image_name,
                        "error": "No valid JSON extracted"
                    }
        else:
            logging.error(f"❌ Failed to get API response for {image_name}")
            return {
                "status": "failed",
                "image_name": image_name,
                "error": "API request failed"
            }

    except Exception as e:
        logging.error(f"❌ Unexpected error processing {image_name}: {str(e)}")
        return {
            "status": "error",
            "image_name": image_name,
            "error": str(e)
        }


def generate_summary_report(processing_results, output_folder):
    """Generate a summary report of the processing"""
    summary = {
        "processing_summary": {
            "total_images": len(processing_results),
            "successful": 0,
            "failed": 0,
            "errors": 0,
            "success_rate": 0
        },
        "processed_files": []
    }
    
    for result in processing_results:
        summary["processed_files"].append(result)
        if result["status"] == "success":
            summary["processing_summary"]["successful"] += 1
        elif result["status"] == "failed":
            summary["processing_summary"]["failed"] += 1
        else:
            summary["processing_summary"]["errors"] += 1
    
    if summary["processing_summary"]["total_images"] > 0:
        success_rate = (summary["processing_summary"]["successful"] / summary["processing_summary"]["total_images"]) * 100
        summary["processing_summary"]["success_rate"] = round(success_rate, 2)
    
    # Save summary report
    summary_file = os.path.join(output_folder, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n📊 Processing Summary saved to: {summary_file}")
    return summary

def process_images_parallel(image_batch, api_key, output_folder, max_workers=5):
    """Process a batch of images in parallel"""
    processing_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures for all images in the batch
        future_to_image = {
            executor.submit(process_single_image, image_path, api_key, output_folder): image_path 
            for image_path in image_batch
        }
         
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result = future.result()
                processing_results.append(result)
            except Exception as exc:
                logging.error(f"{image_path} generated an exception: {exc}")
                processing_results.append({
                    "status": "error",
                    "image_name": os.path.basename(image_path),
                    "error": str(exc)
                })
    
    return processing_results

def main():
    parser = argparse.ArgumentParser(description='Other brands invoice extraction')
    parser.add_argument('--input_folder', required=True, help='Folder with input images')
    parser.add_argument('--output_folder', required=True, help='Folder for output JSON files')
    parser.add_argument('--api_key', required=True, help='Qubrid API key')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    api_key = args.api_key

    if not os.path.exists(input_folder):
        logging.error(f"Error: Input folder not found at {input_folder}")
        return
    
    # Setup logging
    log_file = setup_logging(input_folder)
    logging.info(f"Log file: {log_file}")
    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {output_folder}")
    
    # Test API connection
    logging.info("Testing API connection...")
    if not test_api_connection(api_key):
        logging.error("API connection test failed. Please check your API key and network connection.")
        return
    
    # Get all image files
    image_files = get_image_files(input_folder)
    
    if not image_files:
        logging.error(f"No image files found in {input_folder}")
        return
    
    logging.info(f"Found {len(image_files)} image files to process")
    
    # Filter out images that already have JSON files
    pending_images = get_pending_images(image_files, output_folder)
    
    if not pending_images:
        logging.info("🎉 All images have already been processed!")
        return
    
    logging.info(f"📝 {len(pending_images)} images pending processing (after skipping already processed ones)")
    
    # Display the files that will be processed
    logging.info("Files to be processed:")
    for i, image_file in enumerate(pending_images, 1):
        logging.info(f"  {i}. {os.path.basename(image_file)}")
    
    # Process images in parallel batches
    processing_results = []
    batch_size = 2  # Process 2 images in parallel
    
    for i in range(0, len(pending_images), batch_size):
        batch = pending_images[i:i + batch_size]
        logging.info(f"\n🔄 Processing batch {i//batch_size + 1} with {len(batch)} images...")
        
        batch_results = process_images_parallel(batch, api_key, output_folder, max_workers=batch_size)
        processing_results.extend(batch_results)
        
        logging.info(f"✅ Completed batch {i//batch_size + 1}")
    
    # Generate summary report
    summary = generate_summary_report(processing_results, output_folder)
    
    # Display final summary
    logging.info("\n" + "="*60)
    logging.info("PROCESSING COMPLETED")
    logging.info("="*60)
    logging.info(f"Total images processed: {summary['processing_summary']['total_images']}")
    logging.info(f"Successful: {summary['processing_summary']['successful']}")
    logging.info(f"Failed: {summary['processing_summary']['failed']}")
    logging.info(f"Errors: {summary['processing_summary']['errors']}")
    logging.info(f"Success rate: {summary['processing_summary']['success_rate']}%")
    logging.info(f"JSON files saved to: {output_folder}")
    logging.info(f"Detailed log: {log_file}")

if __name__ == "__main__":
    main()