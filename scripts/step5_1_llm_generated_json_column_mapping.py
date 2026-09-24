


import json
import csv
import os
import glob
import re
import pandas as pd
import sys

def standardize_column_name(col_name):
    """
    Standardize column names by converting to lowercase and replacing spaces/dashes with underscores.
    """
    if not col_name:
        return ""
   
    # Convert to lowercase
    col_name = str(col_name).lower()
   
    # Replace spaces, hyphens, and multiple underscores with single underscore
    col_name = re.sub(r'[\s\-]+', '_', col_name)
    col_name = re.sub(r'_+', '_', col_name)
   
    # Remove any remaining special characters
    col_name = re.sub(r'[^a-z0-9_]', '', col_name)
   
    # Remove leading/trailing underscores
    col_name = col_name.strip('_')
   
    return col_name

def extract_value(data_dict, possible_keys):
    """
    Extract value from dictionary by checking multiple possible keys (exact match only).
    Uses standardization (lowercase, spaces/dashes to underscores) so "no of packs" matches "no_of_packs".
    """
    # First check exact matches
    for key in possible_keys:
        if key in data_dict:
            value = data_dict[key]
            if isinstance(value, (dict, list)):
                return ''
            return value
   
    # Check case-insensitive exact matches
    data_dict_lower = {k.lower(): v for k, v in data_dict.items()}
    for key in possible_keys:
        key_lower = key.lower()
        if key_lower in data_dict_lower:
            value = data_dict_lower[key_lower]
            if isinstance(value, (dict, list)):
                return ''
            return value
   
    # Check standardized match (spaces/dashes -> underscores, strip special chars)
    data_dict_std = {standardize_column_name(k): v for k, v in data_dict.items()}
    for key in possible_keys:
        std_key = standardize_column_name(key)
        if std_key and std_key in data_dict_std:
            value = data_dict_std[std_key]
            if isinstance(value, (dict, list)):
                return ''
            return value
   
    return ''

def number_to_words(num):
    """
    Convert a number to words in Indian numbering system.
    """
    def convert_less_than_one_thousand(n):
        ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
                "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
                "Seventeen", "Eighteen", "Nineteen"]
        tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
       
        if n == 0:
            return ""
       
        if n < 20:
            return ones[n]
        elif n < 100:
            return tens[n // 10] + (" " + ones[n % 10] if n % 10 != 0 else "")
        else:
            return ones[n // 100] + " Hundred" + (" " + convert_less_than_one_thousand(n % 100) if n % 100 != 0 else "")
   
    try:
        if isinstance(num, (int, float)):
            num_str = f"{num:.2f}"
        else:
            num_str = str(num)
           
        num_str = re.sub(r'[^\d.]', '', num_str)
        if not num_str:
            return ""
           
        if '.' in num_str:
            int_part_str, dec_part_str = num_str.split('.')
        else:
            int_part_str, dec_part_str = num_str, "00"
       
        dec_part_str = dec_part_str.ljust(2, '0')[:2]
       
        int_part = int(int_part_str) if int_part_str else 0
        dec_part = int(dec_part_str)
       
        if int_part == 0 and dec_part == 0:
            return "Zero Rupees Only"
       
        words = ""
       
        if int_part > 0:
            crore = int_part // 10000000
            lakh = (int_part % 10000000) // 100000
            thousand = (int_part % 100000) // 1000
            hundred = (int_part % 1000) // 100
            remainder = int_part % 100
           
            if crore > 0:
                words += convert_less_than_one_thousand(crore) + " Crore "
            if lakh > 0:
                words += convert_less_than_one_thousand(lakh) + " Lakh "
            if thousand > 0:
                words += convert_less_than_one_thousand(thousand) + " Thousand "
            if hundred > 0:
                words += convert_less_than_one_thousand(hundred) + " Hundred "
            if remainder > 0:
                words += convert_less_than_one_thousand(remainder)
           
            words = words.strip()
            words += " Rupees"
       
        if dec_part > 0:
            if int_part > 0:
                words += " and "
            words += convert_less_than_one_thousand(dec_part) + " Paisa"
       
        words += " Only"
        return words
       
    except Exception:
        return ""

def json_files_to_csv(folder_path, output_csv_path, mapping_file_path):
    """
    Convert all JSON files in a folder to CSV format, extracting line items
    and keeping only the specified columns.
    """
    # Expected columns in the desired order
    expected_columns = [
        'file_name',
        'invoice_id',
        'invoice_date',
        'supplier_name',
        'description',
        'product_code',
        'quantity',
        'Volume',
        'rate_per_unit',
        'rate_per_ltr',
        'value',
        'amount',
        'Taxable-Amount',
        'Total_Taxable_Amount',
        'total_amount',
        'Total_Amount_in_words'
    ]

    # Load column mapping file (Excel or CSV)
    if mapping_file_path.endswith('.xlsx'):
        mapping_df = pd.read_excel(mapping_file_path)
    else:
        mapping_df = pd.read_csv(mapping_file_path)

    column_mappings = {}
    for _, row in mapping_df.iterrows():
        field = str(row.get("field", "")).strip()
        col = str(row.get("column", "")).strip()
        if not field or not col:
            continue
        column_mappings.setdefault(field, [])
        if col not in column_mappings[field]:
            column_mappings[field].append(col)

    # Ensure all expected columns have an entry (even if empty list)
    for col in expected_columns:
        column_mappings.setdefault(col, [])
    column_mappings.setdefault('unit_price', [])
   
    all_line_items = []
    json_pattern = os.path.join(folder_path, "*.json")
    json_files = glob.glob(json_pattern)
   
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
   
    print(f"Found {len(json_files)} JSON files")
   
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
               
            if 'line_items' in data and isinstance(data['line_items'], list):
                for line_item in data['line_items']:
                    row_data = {}
                    row_data['file_name'] = os.path.basename(json_file)
                   
                    for column in expected_columns:
                        if column == 'file_name':
                            continue
                        if column == 'Total_Amount_in_words':
                            continue
                       
                        if column == 'unit_price':
                            value = extract_unit_price(line_item, column_mappings[column])
                        else:
                            value = extract_value(line_item, column_mappings[column])
                       
                        line_item_only = column in {'quantity', 'Volume', 'unit_price', 'value', 'amount', 'Taxable-Amount', 'Total_Taxable_Amount'}
                        if not value and not line_item_only and 'invoice_details' in data:
                            if column == 'unit_price':
                                value = extract_unit_price(data['invoice_details'], column_mappings[column])
                            else:
                                value = extract_value(data['invoice_details'], column_mappings[column])
                       
                        if not value and not line_item_only:
                            if column == 'unit_price':
                                value = extract_unit_price(data, column_mappings[column])
                            else:
                                value = extract_value(data, column_mappings[column])
                       
                        if column == 'description' and not value:
                            for key, val in line_item.items():
                                if 'description' in key.lower():
                                    if isinstance(val, (dict, list)):
                                        value = ''
                                    else:
                                        value = val
                                    break
                            if not value:
                                for key, val in data.items():
                                    if isinstance(val, dict):
                                        for sub_key, sub_val in val.items():
                                            if 'description' in str(sub_key).lower():
                                                if isinstance(sub_val, (dict, list)):
                                                    value = ''
                                                else:
                                                    value = sub_val
                                                break
                                    elif 'description' in str(key).lower():
                                        if isinstance(val, (dict, list)):
                                            value = ''
                                        else:
                                            value = val
                                        break
                       
                        if isinstance(value, (dict, list)):
                            value = ''
                       
                        if column in ['quantity', 'unit_price', 'value', 'amount', 'Taxable-Amount',
                                     'Total_Taxable_Amount', 'total_amount']:
                            try:
                                if value and isinstance(value, str):
                                    value_clean = re.sub(r'[^\d.]', '', value)
                                    if value_clean:
                                        value_float = float(value_clean)
                                        if value_float.is_integer():
                                            value = int(value_float)
                                        else:
                                            value = round(value_float, 2)
                            except:
                                pass
                       
                        row_data[column] = value if value is not None else ''
                   
                    if row_data.get('amount') in (None, ''):
                        base_val = row_data.get('Taxable-Amount') or row_data.get('value') or ''
                        if base_val != '':
                            try:
                                if isinstance(base_val, str):
                                    base_val = float(re.sub(r'[^\d.]', '', base_val))
                                else:
                                    base_val = float(base_val)
                                amt = base_val * 1.18
                                row_data['amount'] = int(amt) if amt == int(amt) else round(amt, 2)
                            except:
                                row_data['amount'] = ''
                        else:
                            row_data['amount'] = ''
                   
                    total_amount = row_data.get('total_amount', '')
                    if total_amount:
                        try:
                            row_data['Total_Amount_in_words'] = number_to_words(total_amount)
                        except:
                            row_data['Total_Amount_in_words'] = ''
                    else:
                        row_data['Total_Amount_in_words'] = ''
                   
                    all_line_items.append(row_data)
                   
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
   
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=expected_columns)
        writer.writeheader()
        for item in all_line_items:
            writer.writerow(item)
   
    print(f"\nSuccessfully created CSV with {len(all_line_items)} line items")
    print(f"Output file: {output_csv_path}")
    print(f"Columns: {len(expected_columns)}")
    print("\nColumns in output CSV:")
    for i, col in enumerate(expected_columns, 1):
        print(f"{i:2}. {col}")

def extract_unit_price(data_dict, possible_keys):
    """
    Extract unit_price value from dictionary (exact match only).
    Uses standardization so "unit price" matches "unit_price".
    """
    for key in possible_keys:
        if key in data_dict:
            value = data_dict[key]
            if isinstance(value, (dict, list)):
                return ''
            return value
   
    data_dict_lower = {k.lower(): v for k, v in data_dict.items()}
    for key in possible_keys:
        key_lower = key.lower()
        if key_lower in data_dict_lower:
            value = data_dict_lower[key_lower]
            if isinstance(value, (dict, list)):
                return ''
            return value
   
    data_dict_std = {standardize_column_name(k): v for k, v in data_dict.items()
                     if 'gst' not in k.lower() and 'tax' not in k.lower()}
    for key in possible_keys:
        std_key = standardize_column_name(key)
        if std_key and std_key in data_dict_std:
            value = data_dict_std[std_key]
            if isinstance(value, (dict, list)):
                return ''
            return value
   
    return ''

if __name__ == "__main__":
    # Accept command-line arguments: input_folder output_csv mapping_file
    if len(sys.argv) == 4:
        folder_path = sys.argv[1]
        output_csv = sys.argv[2]
        mapping_file = sys.argv[3]
    else:
        # Default values for backward compatibility
        folder_path = r"output_combined"
        output_csv = r"line_items_branded.csv"
        mapping_file = r"D:\Yash\tag_using_llm\P&P Streamline\column_mapping_file\final_column_mapping.xlsx"
        print("Using default parameters. Usage: python script.py <input_folder> <output_csv> <mapping_file>")
    
    json_files_to_csv(folder_path, output_csv, mapping_file)