# -*- coding: utf-8 -*-
"""
Google Document AI Invoice Parser
- Processes images (JPEG/PNG) in a folder using Document AI.
- Extracts highest‑confidence entities and line items.
- Saves results as a CSV file with S. No. column (G1, G2, ...).
"""

import os
import json
import csv
import argparse
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

# ============================================================================
# Helper functions
# ============================================================================
def process_document(file_path, processor_name, client):
    """Send a single image to Document AI and return the response."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        request = documentai.types.ProcessRequest(
            name=processor_name,
            raw_document=documentai.types.RawDocument(
                content=content,
                mime_type='image/jpeg'   # works for JPEG; for PNG change accordingly
            )
        )
        result = client.process_document(request=request)
        return result
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return None

def extract_json_data(document):
    """Convert Document AI Document object to a Python dict."""
    json_str = documentai.Document.to_json(document)
    return json.loads(json_str)

def extract_bounding_box(entity_or_attr):
    """Extract bounding box coordinates from an entity or attribute."""
    bounding_box = ""
    if "pageAnchor" in entity_or_attr:
        page_refs = entity_or_attr["pageAnchor"].get("pageRefs", [])
        if page_refs and "boundingPoly" in page_refs[0]:
            vertices = page_refs[0]["boundingPoly"].get("normalizedVertices", [])
            coordinates = ";".join([f"({v.get('x',0)},{v.get('y',0)})" for v in vertices])
            bounding_box = coordinates
    return bounding_box

def extract_information(json_content, filename):
    """
    Extract invoice‑level and line‑item data from the JSON output.
    Returns a list of dicts (one per line item, with invoice data attached).
    """
    invoice_data = {"file_name": filename}
    line_items = []

    # Track highest‑confidence invoice‑level entities
    highest_confidence_entities = {}

    for entity in json_content.get("entities", []):
        entity_type = entity.get("type", "")
        mention_text = entity.get("mentionText", "")
        confidence = entity.get("confidence", 0.0)

        if entity_type == "line_item":
            # Process line item properties
            line_item = {"file_name": filename}
            highest_confidence_props = {}

            for attr in entity.get("properties", []):
                attr_type = attr.get("type", "")
                attr_value = attr.get("mentionText", "")
                attr_confidence = attr.get("confidence", 0.0)

                if (attr_type not in highest_confidence_props or
                        attr_confidence > highest_confidence_props[attr_type].get(f"{attr_type}_confidence", 0.0)):
                    prop_data = {
                        attr_type: attr_value,
                        f"{attr_type}_confidence": attr_confidence,
                        f"{attr_type}_bounding_box": extract_bounding_box(attr)
                    }
                    highest_confidence_props[attr_type] = prop_data

            # Merge properties into the line item
            for prop in highest_confidence_props.values():
                line_item.update(prop)

            line_item["line_item_bounding_box"] = extract_bounding_box(entity)
            line_item["line_item_confidence"] = confidence
            line_items.append(line_item)

        else:
            # Invoice‑level entity – keep highest confidence
            if (entity_type not in highest_confidence_entities or
                    confidence > highest_confidence_entities[entity_type].get(f"{entity_type}_confidence", 0.0)):
                entity_data = {
                    entity_type: mention_text,
                    f"{entity_type}_confidence": confidence,
                    f"{entity_type}_bounding_box": extract_bounding_box(entity)
                }
                highest_confidence_entities[entity_type] = entity_data

    # Combine invoice‑level data
    for entity_data in highest_confidence_entities.values():
        invoice_data.update(entity_data)

    if not line_items:
        # No line items found – return only invoice data
        return [invoice_data]
    else:
        # Attach invoice data to each line item
        for item in line_items:
            item.update(invoice_data)
        return line_items

def process_json_files(directory):
    """Read all JSON files in the directory, extract data, return a list of dicts."""
    all_data = []
    for filename in os.listdir(directory):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
                extracted = extract_information(json_content, filename)
                if not extracted:
                    all_data.append({"file_name": filename})
                else:
                    all_data.extend(extracted)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            all_data.append({"file_name": filename, "error": str(e)})

    return all_data

def save_to_csv(data, output_file):
    """Write list of dicts to a CSV file with S. No. column starting from G1."""
    if not data:
        return

    # Collect all possible column names
    keys = set()
    for row in data:
        keys.update(row.keys())
    keys = sorted(keys)

    # Insert 'S. No.' as the first column
    if 'S. No.' not in keys:
        keys = ['S. No.'] + list(keys)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for idx, row in enumerate(data, start=1):
            row['S. No.'] = f'G{idx}'
            writer.writerow(row)

# ============================================================================
# Main pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Google Document AI Invoice Parser')
    parser.add_argument('--input_folder', required=True, help='Folder containing invoice images')
    parser.add_argument('--output_csv', required=True, help='Path to output CSV file')
    parser.add_argument('--service_account_json', required=True, help='Path to service account JSON key file')
    args = parser.parse_args()

    # Fixed configuration (these are specific to your project)
    PROJECT_ID = 'invoice-scanner-429009'
    LOCATION = 'us'
    PROCESSOR_ID = '19d2418d93446091'

    # Setup credentials and client
    credentials = service_account.Credentials.from_service_account_file(args.service_account_json)
    client = documentai.DocumentProcessorServiceClient(credentials=credentials)
    processor_name = f'projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}'

    invoices_directory = args.input_folder
    output_csv = args.output_csv

    # Step 1: Process each image to JSON (skip if JSON already exists)
    print("Step 1: Processing images to JSON...")
    for filename in os.listdir(invoices_directory):
        if not filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            continue

        json_filename = os.path.join(invoices_directory, f"output_{filename}.json")
        if os.path.exists(json_filename):
            print(f"Skipping {filename}, JSON already exists.")
            continue

        file_path = os.path.join(invoices_directory, filename)
        result = process_document(file_path, processor_name, client)
        if result:
            json_data = extract_json_data(result.document)
            with open(json_filename, 'w', encoding='utf-8') as out_f:
                json.dump(json_data, out_f, indent=4)
            print(f"Processed and saved JSON for: {filename}")
        else:
            print(f"Failed to process: {filename}")

    # Step 2: Convert all JSON files to a single CSV
    print("\nStep 2: Converting JSON files to CSV...")
    all_data = process_json_files(invoices_directory)
    save_to_csv(all_data, output_csv)
    print(f"Data has been written to {output_csv}")

if __name__ == "__main__":
    main()