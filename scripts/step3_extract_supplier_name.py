
 
import requests
import json
import base64
import re
import os
import logging
import shutil
from datetime import datetime
import glob
from PIL import Image, ImageOps
import io
import csv
import concurrent.futures


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


def extract_company_name_from_logo(image_path, api_key, width_ratio=0.35, height_ratio=0.22):
    """
    Crop the top-left and top-right regions (where logos may reside) and use Qubrid multimodal API
    to read ONLY the company/brand name from each logo.

    Adds robustness by using EXIF-aware transpose. Returns a dict with 'left' and 'right' company strings or None.
    """
    companies = {"left": None, "right": None}
    try:
        # Open the image once
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)  # handle EXIF orientation safely
        w, h = img.size

        # Define crop boxes for left and right
        crop_boxes = {
            "left": (0, 0, int(w * width_ratio), int(h * height_ratio)),
            "right": (int(w * (1 - width_ratio)), 0, w, int(h * height_ratio))
        }

        for side, crop_box in crop_boxes.items():
            logo_crop = img.crop(crop_box)

            # Skip if crop is too small or empty
            if logo_crop.size[0] < 50 or logo_crop.size[1] < 50:
                continue

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
                "Use proper casing (e.g., 'Aditya Birla'). If no clear name or uncertain, return null."
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

            logging.info(f"Sending logo-only request for {side} side to extract company name...")
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code != 200:
                logging.warning(
                    f"Logo API call failed for {side}: {resp.status_code} | {resp.text[:300]}"
                )
                continue

            # Parse response
            raw = resp.json()
            parsed = clean_json_output(raw)
            val = None
            if isinstance(parsed, dict):
                if "company_from_logo" in parsed:
                    val = parsed["company_from_logo"]
            else:
                content = None
                if "choices" in raw and raw["choices"]:
                    content = raw["choices"][0].get("message", {}).get("content")
                if isinstance(content, str):
                    content_parsed = extract_json_from_text(content)
                    if isinstance(content_parsed, dict) and "company_from_logo" in content_parsed:
                        val = content_parsed["company_from_logo"]

            # Fallback
            if val is None and isinstance(parsed, str):
                maybe = extract_json_from_text(parsed)
                if isinstance(maybe, dict) and "company_from_logo" in maybe:
                    val = maybe["company_from_logo"]

            if isinstance(val, str) and val.strip() and val.lower() != "null":
                companies[side] = val.strip()

    except Exception as e:
        logging.warning(f"Logo extraction error for {os.path.basename(image_path)}: {e}")

    # Return dict if any company found, else None
    if companies["left"] or companies["right"]:
        return companies
    else:
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
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    fixed = json_str

    # Replace single quotes with double quotes carefully
    lines = fixed.split('\n')
    fixed_lines = []
    for line in lines:
        line = re.sub(r':\s*\'', ': "', line)
        line = re.sub(r'\'\s*:', '":', line)
        line = re.sub(r'\'(?=\s*[,}])', '"', line)
        line = re.sub(r'(?<=:\s*)\'', '"', line)
        fixed_lines.append(line)
    fixed = '\n'.join(fixed_lines)

    # Remove trailing commas
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)

    # Fix True/False/Null
    fixed = re.sub(r':\s*True\b', ': true', fixed)
    fixed = re.sub(r':\s*False\b', ': false', fixed)
    fixed = re.sub(r':\s*Null\b', ': null', fixed)
    fixed = re.sub(r':\s*NULL\b', ': null', fixed)

    # Remove comments
    fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)

    # Remove extra commas
    fixed = re.sub(r',\s*,', ',', fixed)

    return fixed


def extract_json_from_text(text):
    if not text:
        return None

    clean_text = text.strip()

    if clean_text.startswith('```json'):
        clean_text = clean_text[7:]
    elif clean_text.startswith('```'):
        clean_text = clean_text[3:]
    if clean_text.endswith('```'):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    for start_char in ['{', '[']:
        start_idx = clean_text.find(start_char)
        if start_idx != -1:
            try:
                return json.loads(clean_text[start_idx:])
            except json.JSONDecodeError:
                continue

    for end_char in ['}', ']']:
        end_idx = clean_text.rfind(end_char)
        if end_idx != -1:
            open_char = '{' if end_char == '}' else '['
            start_idx = clean_text.rfind(open_char, 0, end_idx)
            if start_idx != -1:
                try:
                    return json.loads(clean_text[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    continue

    pattern = r'\{[^{}]*?(?:\{[^{}]*\}[^{}]*?)*\}'
    matches = re.findall(pattern, clean_text, re.DOTALL)
    json_candidates = [match for match in matches if len(match) > 20]
    json_candidates.sort(key=len, reverse=True)

    for candidate in json_candidates:
        try:
            cleaned = clean_json_string(candidate)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            fixed = fix_json_string(candidate)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue

    if clean_text.startswith("{") and clean_text.endswith("}"):
        try:
            import ast
            return ast.literal_eval(clean_text)
        except (SyntaxError, ValueError):
            pass

    return text


def clean_json_output(raw_output):
    if not raw_output:
        return None

    if isinstance(raw_output, dict):
        expected_keys = ['company_from_logo']
        if any(key in raw_output for key in expected_keys):
            return raw_output

        content = None
        if 'choices' in raw_output and raw_output['choices']:
            content = raw_output['choices'][0].get('message', {}).get('content', '')

        if not content:
            for key in ['content', 'text', 'result', 'output', 'response']:
                if key in raw_output and isinstance(raw_output[key], str):
                    content = raw_output[key]
                    break

        if not content:
            for value in raw_output.values():
                if isinstance(value, str) and ('{' in value or '[' in value):
                    content = value
                    break

        if content:
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                result = extract_json_from_text(content)
                if isinstance(result, str):
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

        return raw_output

    elif isinstance(raw_output, str):
        return extract_json_from_text(raw_output)

    return raw_output


def save_json_output(data, output_folder, filename):
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


def get_image_files(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    image_files = []

    for extension in image_extensions:
        pattern = os.path.join(folder_path, f"*{extension}")
        files = glob.glob(pattern, recursive=False)
        image_files.extend(files)

    unique_files = list(set(image_files))
    return sorted(unique_files)


def get_pending_images(image_files, output_folder):
    pending_images = []

    for image_path in image_files:
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        json_path = os.path.join(output_folder, f"{base_name}_logo.json")

        if not os.path.exists(json_path):
            pending_images.append(image_path)
        else:
            logging.info(f"⏭️  Skipping {image_name} - JSON already exists")

    return pending_images


def process_single_image(image_path, api_key, output_folder):
    image_name = os.path.basename(image_path)
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Processing: {image_name}")
    logging.info(f"{'=' * 60}")

    try:
        companies = extract_company_name_from_logo(image_path, api_key)

        if companies:
            logging.info(f"✅ Extracted companies: {companies}")

            data = {
                "filename": image_name,
                "companies_from_logo": companies
            }

            base_name = os.path.splitext(image_name)[0]
            json_filename = f"{base_name}_logo.json"

            save_json_output(data, output_folder, json_filename)

            return {
                "status": "success",
                "image_name": image_name,
                "json_filename": json_filename,
                "data": data
            }
        else:
            logging.warning(f"❌ No companies extracted from {image_name}")
            return {
                "status": "failed",
                "image_name": image_name,
                "error": "No companies extracted"
            }

    except Exception as e:
        logging.error(f"❌ Error processing {image_name}: {str(e)}")
        return {
            "status": "error",
            "image_name": image_name,
            "error": str(e)
        }


def generate_summary_report(processing_results, output_folder):
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

    summary_file = os.path.join(output_folder, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"\n📊 Processing Summary saved to: {summary_file}")
    return summary


def generate_companies_csv(output_folder):
    csv_path = os.path.join(output_folder, "companies_from_logos.csv")
    data = []

    json_files = glob.glob(os.path.join(output_folder, "*_logo.json"))

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                js = json.load(f)
                filename = js['filename']
                companies = js.get('companies_from_logo', {})
                company_left = companies.get('left', '') or ''
                company_right = companies.get('right', '') or ''
                data.append({'filename': filename, 'company_left': company_left, 'company_right': company_right})
        except Exception as e:
            logging.warning(f"Error reading {json_path}: {e}")

    if data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['filename', 'company_left', 'company_right'])
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        logging.info(f"CSV saved to {csv_path}")
    else:
        logging.info("No data to write to CSV")

    return csv_path  # return path for later use


def move_images_based_on_csv(csv_path, input_folder, base_dest_root):
    """
    Reads the CSV and moves images from input_folder to subfolders under base_dest_root
    based on the conditions:
    1. Birla White condition -> birla_white
    2. Other brand keywords -> other_brand
    3. Else -> google
    """
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return

    # Define destination folders
    dest_folders = {
        "birla_white": os.path.join(base_dest_root, "birla_white"),
        "other_brand": os.path.join(base_dest_root, "other_brand"),
        "google": os.path.join(base_dest_root, "google")
    }

    # Create destination folders if they don't exist
    for folder in dest_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Read CSV
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        logging.error(f"Failed to read CSV {csv_path}: {e}")
        return

    if not rows:
        logging.warning("CSV is empty, no images to move.")
        return

    # Define keyword lists for condition 2 (case-insensitive, substring match)
    condition2_keywords = [
        "asian", "anpaints", "apl paints", "ap paints", "as paints",
        "birla", "nerolac", "berger"
    ]

    move_summary = {
        "birla_white": [],
        "other_brand": [],
        "google": [],
        "not_found": [],
        "errors": []
    }

    for row in rows:
        filename = row.get('filename', '').strip()
        if not filename:
            logging.warning("Empty filename in CSV row, skipping.")
            continue

        company_left = (row.get('company_left', '') or '').strip()
        company_right = (row.get('company_right', '') or '').strip()

        # Build full source path
        src_path = os.path.join(input_folder, filename)
        if not os.path.exists(src_path):
            logging.warning(f"Image not found: {src_path}")
            move_summary["not_found"].append(filename)
            continue

        # Determine destination folder
        dest_folder = None

        # Condition 1: Birla White special case
        if ("birla white" in company_right.lower() and
            ("aditya birla" in company_left.lower() or "ultratech" in company_left.lower())):
            dest_folder = dest_folders["birla_white"]
            category = "birla_white"

        # Condition 2: Other brand keywords in company_left
        else:
            found_keyword = False
            company_left_lower = company_left.lower()
            for kw in condition2_keywords:
                if kw in company_left_lower:
                    found_keyword = True
                    break
            if found_keyword:
                dest_folder = dest_folders["other_brand"]
                category = "other_brand"
            else:
                dest_folder = dest_folders["google"]
                category = "google"

        # Move the file
        try:
            dest_path = os.path.join(dest_folder, filename)
            # Avoid overwriting: if dest exists, add a suffix
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(os.path.join(dest_folder, f"{base}_{counter}{ext}")):
                    counter += 1
                dest_path = os.path.join(dest_folder, f"{base}_{counter}{ext}")

            shutil.move(src_path, dest_path)
            logging.info(f"Moved: {filename} -> {category}")
            move_summary[category].append(filename)
        except Exception as e:
            logging.error(f"Failed to move {filename}: {e}")
            move_summary["errors"].append({"filename": filename, "error": str(e)})

    # Write move summary report
    summary_file = os.path.join(base_dest_root, "image_move_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(move_summary, f, indent=2, ensure_ascii=False)

    logging.info("\n" + "=" * 60)
    logging.info("IMAGE MOVING COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Birla White folder: {len(move_summary['birla_white'])} images")
    logging.info(f"Other brand folder: {len(move_summary['other_brand'])} images")
    logging.info(f"Google folder: {len(move_summary['google'])} images")
    logging.info(f"Not found: {len(move_summary['not_found'])} images")
    logging.info(f"Errors: {len(move_summary['errors'])}")
    logging.info(f"Move summary saved to: {summary_file}")


def main():
    input_folder = r"D:\Yash\tag_using_llm\P&P Streamline\all_invoices"
    output_folder = r"D:\Yash\tag_using_llm\P&P Streamline\all_invoices\company_name"

    # Destination root for categorized invoices
    dest_root = r"D:\Yash\tag_using_llm\P&P Streamline\invoices_for_all"

    api_key = "k_e5464b64cc97.3vmxz4fSnLXNOdSDF-yP8M9Fq0ocESZFjKeoGm58xbaA4aztCZu-IA"
    if not api_key:
        logging.error("Error: API key not set")
        return

    if not os.path.exists(input_folder):
        logging.error(f"Error: Input folder not found at {input_folder}")
        return

    log_file = setup_logging(input_folder)
    logging.info(f"Log file: {log_file}")
    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {output_folder}")

    image_files = get_image_files(input_folder)

    if not image_files:
        logging.error(f"No image files found in {input_folder}")
        return

    logging.info(f"Found {len(image_files)} image files to process")

    pending_images = get_pending_images(image_files, output_folder)

    processing_results = []
    if pending_images:
        logging.info(f"📝 {len(pending_images)} images pending processing")

        logging.info("Files to be processed:")
        for i, image_file in enumerate(pending_images, 1):
            logging.info(f"  {i}. {os.path.basename(image_file)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_single_image, image_path, api_key, output_folder) for image_path in pending_images]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                processing_results.append(result)
    else:
        logging.info("🎉 All images have already been processed!")

    summary = generate_summary_report(processing_results, output_folder)

    # Generate CSV and get its path
    csv_path = generate_companies_csv(output_folder)

    # Move images based on CSV conditions
    move_images_based_on_csv(csv_path, input_folder, dest_root)

    logging.info("\n" + "=" * 60)
    logging.info("PROCESSING COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Total images processed: {summary['processing_summary']['total_images']}")
    logging.info(f"Successful: {summary['processing_summary']['successful']}")
    logging.info(f"Failed: {summary['processing_summary']['failed']}")
    logging.info(f"Errors: {summary['processing_summary']['errors']}")
    logging.info(f"Success rate: {summary['processing_summary']['success_rate']}%")
    logging.info(f"JSON files saved to: {output_folder}")
    logging.info(f"Detailed log: {log_file}")


if __name__ == "__main__":
    main()