
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SKU Mapping and Invoice Processing Pipeline
Converts OCR output to structured product data, matches with master SKU list,
performs price/quantity/amount corrections, and exports final data file.
"""

import os
import re
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker

# Optional: download nltk data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

warnings.filterwarnings("ignore")

# ============================================================================
# Parse command-line arguments
# ============================================================================
parser = argparse.ArgumentParser(description='SKU Mapping and Invoice Processing')
parser.add_argument('--input_csv', required=True, help='Path to the input CSV file (line items)')
args = parser.parse_args()

OCR_INPUT_CSV = args.input_csv   # replaces hardcoded 'line_items_branded.csv'

# ================================
# Configuration: File Paths
# ================================
WORKING_DIR = os.getcwd()  # or set to your project root

# Mapping files
SKU_MAPPING_DIR = "SKU_mapping"
CORRECTIONS_FILE = os.path.join(SKU_MAPPING_DIR, "sku_mapping_dictionaries.xlsx")
KEYWORDS_FILE = os.path.join(SKU_MAPPING_DIR, "Keywords.xlsx")
BRAND_DICT_FILE = os.path.join(SKU_MAPPING_DIR, "product_brand_dict.xlsx")
COLOUR_MAPPING_FILE = os.path.join(SKU_MAPPING_DIR, "colour_mapping_dict.xlsx")
SHORT_FORM_DICT_FILE = os.path.join(SKU_MAPPING_DIR, "short_form_dictionary.xlsx")
AVG_PRICE_FILE = "Paints CSV Report - May'25.xlsx"
SKU_MASTER_FILE = os.path.join(SKU_MAPPING_DIR, "Paint - SKU Master - 06-01-2026.xlsx")

# Input/Output files
RAW_SKU_EXCEL = 'Rawsku.xlsx'
PROCESSED_DESC_EXCEL = os.path.join(SKU_MAPPING_DIR, 'processed_product_descriptions.xlsx')
HIGH_FREQ_OUTPUT = os.path.join(SKU_MAPPING_DIR, 'untagged_high_frequency_words.xlsx')
SUPPLIER_RE_FILE = os.path.join(SKU_MAPPING_DIR, "Supplier_RE.xlsx")
CORRECTED_FILE_CSV = os.path.join(SKU_MAPPING_DIR, "Corrected File.csv")
MASTER_MATCHING_CSV = os.path.join(SKU_MAPPING_DIR, "master_matching.csv")
PS_MATCHED_OUTPUT = os.path.join(SKU_MAPPING_DIR, "PS_matched_output.xlsx")
POST_PROCESSING_DIR = "post_processing files"
CORRECTIONS_COMPLETE_CSV = "Corrections complete.csv"
INTERMEDIATE_FILE_CSV = "Intermediate file.csv"
REFER_DOC_XLSX = "Refer Doc.xlsx"
DATA_FILE_XLSX = "Data_File_by_llm.xlsx"

# Ensure directories exist
os.makedirs(SKU_MAPPING_DIR, exist_ok=True)
os.makedirs(POST_PROCESSING_DIR, exist_ok=True)

# ================================
# Load all mapping data (global)
# ================================
print("Loading mapping files...")

# Sheets from sku_mapping_dictionaries.xlsx
corrections_df = pd.read_excel(CORRECTIONS_FILE, sheet_name="corrections")
high_freq_series_df = pd.read_excel(CORRECTIONS_FILE, sheet_name="high_freq_series_names")
unit_map_df = pd.read_excel(CORRECTIONS_FILE, sheet_name="unit_map")
unit_list_df = pd.read_excel(CORRECTIONS_FILE, sheet_name="unit_list")
series_list_df = pd.read_excel(CORRECTIONS_FILE, sheet_name="series_list")
brand_supplier_df = pd.read_excel(CORRECTIONS_FILE, sheet_name="brand_supplier_list")
brand_var_df = pd.read_excel(CORRECTIONS_FILE, sheet_name="brands_var_list")

# Sheets from Keywords.xlsx
product_keywords_df = pd.read_excel(KEYWORDS_FILE, sheet_name="Product Final")
brand_keywords_df = pd.read_excel(KEYWORDS_FILE, sheet_name="Brand Final")
not_paints_keyword = list(pd.read_excel(KEYWORDS_FILE, sheet_name="not_paint")["Not Paint Keyword"])
common_keywords_df = pd.read_excel(KEYWORDS_FILE, sheet_name="common_keywords")
common_keywords_list = common_keywords_df["Words"].tolist()

# Other mapping files
brand_dict_df = pd.read_excel(BRAND_DICT_FILE, sheet_name="Brand Name Mapping")
color_list_df = pd.read_excel(COLOUR_MAPPING_FILE, sheet_name="Colour List")
color_mapping_df = pd.read_excel(COLOUR_MAPPING_FILE, sheet_name="Colour Dictionary")
product_brand_mapping_df = pd.read_excel(BRAND_DICT_FILE, sheet_name="Product Brand Mapping")
short_form_dict_df = pd.read_excel(SHORT_FORM_DICT_FILE)

# Average price and master SKU
avg_price_df = pd.read_excel(AVG_PRICE_FILE, sheet_name='Sheet2')
size_df_raw = pd.read_excel(SKU_MASTER_FILE, sheet_name='FInal SKu')
size_df_raw = size_df_raw.drop(columns=['Period', 'Comment', 'Price Segment'])

# ================================
# Preprocess master SKU dataframe
# ================================
def split_and_strip(input_string):
    parts = input_string.split('-')
    if re.search(r'\d', parts[-1]):
        parts.pop()
    result = '-'.join(parts)
    return result

master_file = size_df_raw.drop(columns='Base Name')
master_file = master_file.rename(columns={
    "Droid SKU New Name Dec'25": 'SKU Name',
    "Company Name": 'Brand Name',
    "Brand Name": 'Series Name',
    "Category - MP": 'Product'
})
master_file['SKU Name'] = master_file["SKU Name"].apply(split_and_strip)
master_file['Series Name'] = master_file['Series Name'].str.upper()
master_file['Brand Name'] = master_file['Brand Name'].replace({'Asian Paints': 'asianpaints'})
master_file['Brand Name'] = master_file[['SKU Name', 'Brand Name']].apply(
    lambda x: 'British Paints' if 'British Paints' in x[0] else x[1], axis=1
)
master_df = master_file[['SKU Name', 'Pack Size']]

size_df = size_df_raw[["Droid SKU New Name Dec'25", "Pack Size", "Company Name"]]
size_df = size_df[size_df['Company Name'] != 'Other Brands']
size_df = size_df.rename(columns={"Droid SKU New Name Dec'25": 'SKU Name with packsize'})
size_df['SKU Name'] = size_df['SKU Name with packsize'].apply(
    lambda x: split_and_strip(x) if 'Tools for Paints' not in x else x
)
size_df['Final Name'] = size_df['SKU Name with packsize']
size_df['Pack Size'] = size_df['Pack Size'].astype(str)
size_df['unit_only'] = size_df['Pack Size'].str.replace(r'[^a-zA-Z]', '', regex=True)
size_df['packsize_only'] = size_df['Pack Size'].str.extract(r'(\d*\.?\d+)')

# Build dictionaries and lists for parsing
units_list = unit_list_df["Units"].to_list()
unit_map = (unit_map_df.set_index("Unit").to_dict())["Standardized Unit"]
high_freq_series_names = high_freq_series_df["Series"].tolist()
corrections = (corrections_df.set_index("Expression").to_dict())["Correction"]
product_keywords = product_keywords_df.groupby("Product")['Keyword'].unique().to_dict()
brand_keywords = brand_keywords_df.groupby("Brand")['Keyword'].unique().to_dict()
brand_dict = brand_dict_df.groupby("Brand")['Keyword'].unique().to_dict()
color_list = color_list_df["Colours"].to_list()
color_mapping = (color_mapping_df.set_index("Short Form").to_dict())["Colour"]
product_brand_mapping = product_brand_mapping_df.groupby("Product")["Brand"].unique().to_dict()
short_form_dict = (short_form_dict_df.set_index("Short Form").to_dict())["Expansion"]
brand_supplier = brand_supplier_df["Brands"].to_list()
brands_var_lst = brand_var_df["Brands"].to_list()
series_list = series_list_df['Series'].to_list()

# Build reverse mapping for product->brands
product_brand_reverse = {}
for product_name, brands in product_brand_mapping.items():
    for brand in brands:
        product_brand_reverse.setdefault(product_name, []).append(brand)

# Units sorted by length descending
units_sorted = sorted(units_list, key=len, reverse=True)
pack_size_units_pattern = r'(?:^|(?<=[\s\-_]))[\(]*\s*(\d+(?:\.\d+)?(?:\s*[xX*]\s*\d+(?:\.\d+)?)?)(?:\s*[-/_]?\s*)(' + '|'.join(re.escape(u) for u in units_sorted) + r')\b'

# ================================
# Helper functions for parsing
# ================================
def parse_product_description(description):
    """Parse raw product description into brand, series, product, color, pack size, units."""
    # Initialize
    brand = 'Unknown'
    series_name = 'Unknown'
    product_name = 'Unknown'
    color = 'Unknown'
    pack_size = 'Unknown'
    units = 'Unknown'
    remaining_description = ''

    if pd.isna(description):
        description = ''
    elif not isinstance(description, str):
        description = str(description)

    description = description.lower()
    token = word_tokenize(description)
    token_series = pd.Series(token)
    token_series = token_series.replace(short_form_dict)
    token = list(token_series)
    description = " ".join(token)

    desc = description.upper()
    desc = re.sub(r'[^\x00-\x7F]+', ' ', desc)

    for pattern, replacement in corrections.items():
        desc = re.sub(pattern, replacement, desc)
    desc = re.sub(r'\s+', ' ', desc).strip()

    for abbr, full_name in color_mapping.items():
        desc = re.sub(r'\b' + re.escape(abbr) + r'\b', full_name, desc)

    # Extract pack size and units (last match)
    pack_size_matches = list(re.finditer(pack_size_units_pattern, desc))
    pack_size_match = pack_size_matches[-1] if pack_size_matches else None
    if pack_size_match:
        pack_size = pack_size_match.group(1).replace(' ', '')
        units = pack_size_match.group(2)
        desc = desc.replace(pack_size_match.group(0), '').strip()

    # Extract color
    for color_option in color_list:
        if re.search(r'\b' + re.escape(color_option.upper()) + r'\b', desc):
            color = color_option
            if color_option != 'WHITE':
                desc = re.sub(r'\b' + re.escape(color_option.upper()) + r'\b', '', desc).strip()
            break

    # Extract series and brand
    series_found = False
    for brand_key, series_list_val in brand_keywords.items():
        for series in series_list_val:
            if re.search(r'\b' + re.escape(series.upper()) + r'\b', desc, flags=re.IGNORECASE):
                series_name = series
                brand = brand_key
                if series_name.lower() not in common_keywords_list:
                    desc = re.sub(r'\b' + re.escape(series.upper()) + r'\b', '', desc).strip()
                series_found = True
                break
        if series_found:
            break
    # Special case for Birla
    excluded_words = ['Opus', 'OPUS', 'opus', 'White', 'white', 'WHITE', 'HIL', 'Hil', 'hil']
    pattern_birla = rf'\bBirla\b(?!\s+({"|".join(excluded_words)}))'
    if re.search(pattern_birla, desc, flags=re.IGNORECASE):
        brand = "Birla HIL_Birla Opus_Birla White"

    # If brand still unknown, use brand_dict
    if brand == 'Unknown':
        brand_found = False
        for full_name, abbreviations in brand_dict.items():
            for abbr in abbreviations:
                if re.search(r'\b' + re.escape(abbr.upper()) + r'\b', desc, flags=re.IGNORECASE):
                    brand = full_name
                    if brand.lower() not in common_keywords_list:
                        desc = re.sub(r'\b' + re.escape(abbr.upper()) + r'\b', '', desc).strip()
                    brand_found = True
                    break
            if brand_found:
                break

    # Extract product name
    product_name_found = False
    for product, keywords in product_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.upper()) + r'\b', desc, flags=re.IGNORECASE):
                product_name = product
                desc = re.sub(r'\b' + re.escape(keyword.upper()) + r'\b', '', desc, flags=re.IGNORECASE).strip()
                product_name_found = True
                break
        if product_name_found:
            break

    # Assign brand based on product if not found
    if brand == 'Unknown' and product_name != 'Unknown':
        possible_brands = product_brand_reverse.get(product_name, [])
        if len(possible_brands) == 1:
            brand = possible_brands[0]
        elif len(possible_brands) > 1:
            brand_match = process.extractOne(desc, possible_brands, scorer=fuzz.token_sort_ratio)
            if brand_match and brand_match[1] > 85:
                brand = brand_match[0]

    remaining_description = ' '.join(desc.split())

    return {
        'Brand': brand,
        'Series Name': series_name,
        'Product': product_name,
        'Color': color,
        'Pack Size': pack_size,
        'Units': units,
        'Remaining Description': remaining_description
    }

def extract_high_frequency_words(descriptions, frequency_threshold=10):
    """Extract high-frequency words not present in tagging dictionaries."""
    combined_text = ' '.join(descriptions).upper()
    combined_text = re.sub(r'[^\x00-\x7F]+', ' ', combined_text)
    combined_text = re.sub(r'[^\w\s]', ' ', combined_text)
    combined_text = re.sub(r'\d+', ' ', combined_text)
    words = combined_text.split()
    word_counts = Counter(words)
    high_freq_words = {w: c for w, c in word_counts.items() if c >= frequency_threshold}

    # Collect tagged words from dictionaries
    tagged_words = set()
    for pattern in corrections:
        word = re.sub(r'[\\\^\$\.\|\?\*\+\(\)\[\]\{\}]', '', pattern)
        word = word.replace('\\b', '').replace('\\', '').strip()
        if word:
            tagged_words.add(word.upper())
    for brand, abbrs in brand_dict.items():
        tagged_words.add(brand.upper())
        for abbr in abbrs:
            tagged_words.add(abbr.upper())
    for product, keywords in product_keywords.items():
        tagged_words.add(product.upper())
        for kw in keywords:
            tagged_words.add(kw.upper())
    for color in color_list:
        tagged_words.add(color.upper())
    for abbr, full in color_mapping.items():
        tagged_words.add(abbr.upper())
        tagged_words.add(full.upper())
    for brand, series_list_val in brand_keywords.items():
        for series in series_list_val:
            tagged_words.add(series.upper())

    untagged = {w: c for w, c in high_freq_words.items() if w not in tagged_words}
    df_out = pd.DataFrame(list(untagged.items()), columns=['Word', 'Frequency'])
    df_out = df_out.sort_values(by='Frequency', ascending=False)
    return df_out

def process_excel_file(input_df, output_file, high_freq_output_file):
    """Read input dataframe, parse descriptions, save results."""
    df = input_df.copy()
    if 'Description' not in df.columns:
        print("Input must have 'Description' column.")
        return
    df['Description'] = df['Description'].astype(str).fillna('')
    untagged_words = extract_high_frequency_words(df['Description'])
    untagged_words.to_excel(high_freq_output_file, index=False)
    parsed = df['Description'].apply(parse_product_description)
    parsed_df = pd.DataFrame(parsed.tolist())
    result_df = pd.concat([df, parsed_df], axis=1)
    result_df.to_excel(output_file, index=False)
    print(f"Processed descriptions saved to {output_file}")

# ================================
# Step 1: Load OCR input and parse descriptions
# ================================
print("\n=== Step 1: Loading OCR output and parsing descriptions ===")
ocr_output = pd.read_csv(OCR_INPUT_CSV)


input_df = ocr_output[['S. No.', 'description', 'supplier_name']].rename(columns={'description': 'Description'})
input_df.to_excel(RAW_SKU_EXCEL, index=False)

process_excel_file(input_df, PROCESSED_DESC_EXCEL, HIGH_FREQ_OUTPUT)

# ================================
# Step 2: Handle unknown brands using supplier mapping
# ================================
print("\n=== Step 2: Supplier-based brand correction ===")
df = pd.read_excel(PROCESSED_DESC_EXCEL)
known_df = df[df['Brand'] != "Unknown"]
unknown_df = df[df['Brand'] == "Unknown"]

re_df = pd.read_excel(SUPPLIER_RE_FILE)
re_dict = re_df.groupby('Supplier Name RE')['Brand'].unique().to_dict()

def get_brand(supplier_nm):
    if not isinstance(supplier_nm, str):
        return supplier_nm
    for br in re_dict:
        if br in supplier_nm.lower():
            brands = re_dict[br]
            if len(brands) > 1:
                return "_".join(brands[:3])
            else:
                return brands[0]
    return supplier_nm

unknown_df['supplier_name_clean'] = unknown_df['supplier_name'].apply(get_brand)
known_df['supplier_name_clean'] = known_df['supplier_name'].apply(get_brand)

# British Paints override
for col in ['supplier_name_clean']:
    unknown_df[col] = unknown_df.apply(lambda x: 'British Paints' if 'british paints division' in str(x['supplier_name']).lower() else x[col], axis=1)
    known_df[col] = known_df.apply(lambda x: 'British Paints' if 'british paints division' in str(x['supplier_name']).lower() else x[col], axis=1)

unknown_df['Brand'] = unknown_df.apply(lambda z: z['supplier_name_clean'] if z['supplier_name_clean'] in brand_supplier else 'Unknown', axis=1)

df = pd.concat([known_df, unknown_df], ignore_index=True)

# Split multi-brand entries
rem_df = df[df['Brand'] != "Birla HIL_Birla Opus_Birla White"]
split_df = df[df['Brand'] == "Birla HIL_Birla Opus_Birla White"]
birla_parts = []
for rpt in range(3):
    temp = split_df.copy()
    temp['Brand'] = "Birla HIL_Birla Opus_Birla White".split("_")[rpt]
    birla_parts.append(temp)
concat_df = pd.concat(birla_parts, ignore_index=True)
brand_df = pd.concat([rem_df, concat_df], ignore_index=True)

rem_df = brand_df[brand_df['Brand'] != "JK Cement_JK MAXX Paints_JK Lakshmi"]
split_df = brand_df[brand_df['Brand'] == "JK Cement_JK MAXX Paints_JK Lakshmi"]
jk_parts = []
for rpt in range(3):
    temp = split_df.copy()
    temp['Brand'] = "JK Cement_JK MAXX Paints_JK Lakshmi".split("_")[rpt]
    jk_parts.append(temp)
concat_df = pd.concat(jk_parts, ignore_index=True)
brand_df = pd.concat([rem_df, concat_df], ignore_index=True)

# Flag non-paint products
def product_check(pr_name):
    if isinstance(pr_name, str):
        for kw in not_paints_keyword:
            if kw in pr_name.lower():
                return 1
    return 0

brand_df['Not Paint Product Flag'] = brand_df['Description'].apply(product_check)
paints_df = brand_df[brand_df['Not Paint Product Flag'] == 0]

# Final brand cleanup
paints_df['Brand'] = paints_df['Brand'].replace({"Unknown": "Other Brands", "OTHER BRANDS": "Other Brands"})
all_brands_df = pd.concat([paints_df, known_df], ignore_index=True)
brand_supplier_ls_refined = list(set(brand_supplier) - {"Birla HIL_Birla Opus_Birla White", 'JK Cement_JK MAXX Paints_JK Lakshmi'})
all_brands_df['Brand'] = all_brands_df.apply(lambda z: z['supplier_name_clean'] if (z['supplier_name_clean'] != z['Brand'] and z['supplier_name_clean'] in brand_supplier_ls_refined and z['Brand'] != 'British Paints') else z['Brand'], axis=1)

all_brands_df.to_csv(CORRECTED_FILE_CSV, index=False)

df = all_brands_df.drop_duplicates().dropna(subset=['Description'])
df["Brand"] = df["Brand"].str.title()
df['Brand'] = df['Brand'].replace({
    "Asian Paints": "asianpaints", "Jsw": "JSW", "Jk Cement": "JK Cement",
    "Asianpaints": "asianpaints", "Dr. Fixit": "DR. FIXIT", 'Birla Hil': "Birla HIL",
    "Jk Maxx Paints": "JK MAXX Paints", 'Jk Cement ': 'JK Cement', 'Mrf': 'MRF'
})

# ================================
# Step 3: Prepare master database for fuzzy matching
# ================================
print("\n=== Step 3: Preparing master SKU database ===")
db_df = master_file.dropna(subset=['SKU Name']).copy()

def sku_refiner(row):
    sku = row['SKU Name']
    brand = row['Brand Name']
    sku_cleaned = sku.replace("-", "").replace(brand, "").strip().lower()
    sku_cleaned = re.sub(" +", " ", sku_cleaned)
    sku_cleaned = sku_cleaned.replace("(cw tinting base)", "")
    return sku_cleaned

db_df['SKU refined'] = db_df.apply(sku_refiner, axis=1)
db_df = db_df[['Brand Name', 'SKU Name', 'SKU refined', 'Product', 'Series Name']].drop_duplicates()
db_df.rename(columns={"Brand Name": "Brand"}, inplace=True)

# Clean description column
df['Desc_cleaned'] = df['Description'].str.lower()
df['Desc_cleaned'] = df['Desc_cleaned'].apply(lambda z: re.sub(r"\d", " ", z))
df['Desc_cleaned'] = df['Desc_cleaned'].apply(lambda z: re.sub(r"[^a-z\s]", " ", z))
df['Desc_cleaned'] = df['Desc_cleaned'].apply(lambda z: re.sub(" +", " ", z))

stop_words = list(stopwords.words('english'))
stop_words.extend(['ml', 'ltr', 'lt', 'kg', 'l', "buy", "for", "price", "litre", "at", "from",
                   "india", "best", "prices", "online", "get", 'gst', 'hi', 'cgst', 'bag', 'gms',
                   'central', 'state', 'ap'])

def cleaner(txt_str):
    tokens = word_tokenize(txt_str)
    tokens = [t for t in tokens if t not in stop_words]
    tokens_series = pd.Series(tokens).replace(short_form_dict)
    tokens = list(tokens_series)
    tokens = [z for z in tokens if len(z) >= 2]
    cleaned_str = " ".join(tokens)
    fin_str = cleaned_str
    for b in brands_var_lst:
        if cleaned_str.startswith(b):
            fin_str = re.sub(b, "", cleaned_str, count=1)
    return fin_str.strip()

df['Desc_cleaned'] = df['Desc_cleaned'].apply(cleaner)
df = df[df['Desc_cleaned'] != ""]

# ================================
# Step 4: Fuzzy matching to SKU refined
# ================================
print("\n=== Step 4: Fuzzy matching descriptions to SKU refined ===")
def matcher_brand_specific_all(string, brand, product, series):
    if brand != "Unknown":
        brand_df = db_df[db_df['Brand'] == brand]
        if series in series_list:
            brand_df = brand_df[brand_df['Series Name'].str.contains(series, na=False)]
        if product != "Unknown":
            if product == "Emulsion":
                brand_df = brand_df[(brand_df['Product'] == "Interior Emulsion") | (brand_df['Product'] == "Exterior Emulsion")]
            elif product == "Putty":
                brand_df = brand_df[(brand_df['Product'] == "Cement Putty") | (brand_df['Product'] == "Acrylic Putty")]
            else:
                brand_df = brand_df[brand_df['Product'] == product]
    else:
        brand_df = db_df
        if product != "Unknown":
            if product == "Emulsion":
                brand_df = brand_df[(brand_df['Product'] == "Interior Emulsion") | (brand_df['Product'] == "Exterior Emulsion")]
            elif product == "Putty":
                brand_df = brand_df[(brand_df['Product'] == "Cement Putty") | (brand_df['Product'] == "Acrylic Putty")]
            else:
                brand_df = brand_df[brand_df['Product'] == product]
    choices = list(brand_df['SKU refined'].unique())
    if not choices:
        return (-1, -1)
    match = process.extractOne(string, choices, scorer=fuzz.token_sort_ratio)
    if match:
        return (match[0], match[1])
    return (-1, -1)

df[['SKU refined', 'Match Score']] = df.apply(
    lambda s: matcher_brand_specific_all(s['Desc_cleaned'], s['Brand'], s['Product'], s['Series Name']),
    axis=1, result_type='expand'
)

db_df_unique = db_df.drop_duplicates(subset=['SKU Name'])
df_merged = pd.merge(df, db_df_unique[['SKU refined', 'SKU Name', 'Brand']], on=['SKU refined', 'Brand'], how='left')
df_merged = df_merged.sort_values("Match Score", ascending=False).drop_duplicates(subset=['S. No.'], keep='first')
df_merged['Brand'] = df_merged['Brand'].replace({'British Paints': "Berger"})
df_merged.to_csv(MASTER_MATCHING_CSV, index=False)

# ================================
# Step 5: Pack size normalization and matching
# ================================
print("\n=== Step 5: Pack size normalization and matching ===")
def normalize_and_convert_units(value, unit):
    if pd.isna(value) or pd.isna(unit):
        return None, 'Unknown'
    unit = unit.strip().upper()
    unit = unit_map.get(unit, unit)
    try:
        val = float(str(value).strip())
    except:
        val = None
    if val is None or unit == 'Unknown':
        return None, 'Unknown'
    if unit == 'ML':
        val = val / 1000.0
        unit = 'L'
    elif unit == 'G':
        val = val / 1000.0
        unit = 'KG'
    return val, unit

def parse_pack_size(pack_size):
    if pd.isna(pack_size) or not isinstance(pack_size, str):
        return None, 'Unknown'
    parts = pack_size.strip().split()
    if len(parts) == 2:
        val_str, unit_str = parts
        try:
            val = float(val_str)
        except:
            val = None
        return val, unit_str
    else:
        match = re.match(r"^(\d+(\.\d+)?)(.*)$", pack_size.strip())
        if match:
            val = float(match.group(1))
            unit_str = match.group(3).strip()
            return val, unit_str if unit_str else 'Unknown'
    return None, 'Unknown'

def load_and_prepare_data(master_df, source_path):
    source_df = pd.read_csv(source_path)
    master_df[['Raw_Pack_Num', 'Raw_Unit']] = master_df.apply(lambda r: pd.Series(parse_pack_size(r['Pack Size'])), axis=1)
    master_df[['Pack Size Num', 'Norm Units']] = master_df.apply(lambda r: pd.Series(normalize_and_convert_units(r['Raw_Pack_Num'], r['Raw_Unit'])), axis=1)
    master_df['SKU_lower'] = master_df['SKU Name'].str.lower().str.strip()
    source_df[['Pack Size Num', 'Norm Units']] = source_df.apply(lambda r: pd.Series(normalize_and_convert_units(r['Pack Size'], r['Units'])), axis=1)
    source_df['SKU_lower'] = source_df['SKU Name'].str.lower().str.strip()
    master_df['Pack_Full'] = master_df.apply(lambda r: f"{r['Pack Size Num']:.6g} {r['Norm Units']}" if r['Pack Size Num'] is not None and r['Norm Units'] != 'Unknown' else "Unknown", axis=1)
    source_df['Pack_Full'] = source_df.apply(lambda r: f"{r['Pack Size Num']:.6g} {r['Norm Units']}" if r['Pack Size Num'] is not None and r['Norm Units'] != 'Unknown' else "Unknown", axis=1)
    return master_df, source_df

def fuzzy_match_sku_and_pack(source_row, master_df):
    source_sku = source_row['SKU_lower']
    pack_num = source_row['Pack Size Num']
    norm_unit = source_row['Norm Units']
    all_skus = master_df['SKU_lower'].unique()
    if len(all_skus) == 0 or not isinstance(source_sku, str):
        return None, None
    result = process.extractOne(source_sku, all_skus, scorer=fuzz.WRatio)
    if not result or result[1] < 60:
        return None, None
    best_sku = result[0]
    candidate_skus = master_df[master_df['SKU_lower'] == best_sku].copy()
    if candidate_skus.empty:
        return None, None
    
    # If we have pack_num, try to find closest pack size
    if pack_num is not None and not candidate_skus['Pack Size Num'].isna().all():
        # Drop rows with NaN pack size
        candidate_skus = candidate_skus.dropna(subset=['Pack Size Num'])
        if not candidate_skus.empty:
            candidate_skus['diff'] = (candidate_skus['Pack Size Num'] - pack_num).abs()
            best_idx = candidate_skus['diff'].idxmin()
            # Check if best_idx is valid (not NaN)
            if pd.notna(best_idx):
                best_match = candidate_skus.loc[best_idx]
                return best_match['SKU Name'], best_match['Pack_Full']
    
    # Fallback: take first candidate
    best_match = candidate_skus.iloc[0]
    return best_match['SKU Name'], None

def match_source_to_master(master_df, source_df):
    merged = pd.merge(source_df, master_df, left_on=['SKU_lower', 'Pack_Full'], right_on=['SKU_lower', 'Pack_Full'], how='left', suffixes=('', '_master'))
    merged['Matched SKU'] = None
    merged['Matched Pack_Full'] = None
    no_match = merged[merged['SKU Name_master'].isna()].copy()
    if not no_match.empty:
        no_match[['Matched SKU', 'Matched Pack_Full']] = no_match.apply(lambda r: pd.Series(fuzzy_match_sku_and_pack(r, master_df)), axis=1)
        merged.update(no_match[['Matched SKU', 'Matched Pack_Full']])
    merged['Final SKU Name'] = merged['Matched SKU'].fillna(merged['SKU Name_master'])
    merged['Final Pack Size'] = merged['Matched Pack_Full'].fillna(merged['Pack_Full'])
    final_df = merged[['S. No.', 'SKU Name', 'Pack Size', 'Units', 'Final SKU Name', 'Final Pack Size']]
    return final_df

master_df, source_df = load_and_prepare_data(master_df, MASTER_MATCHING_CSV)
final_result = match_source_to_master(master_df, source_df)

def _choose_pack_size(row):
    orig = row['Pack Size']
    if pd.isna(orig) or str(orig).strip() in ('', 'Unknown', 'nan', 'NaN', 'None'):
        return 'Unknown'
    final = row['Final Pack Size']
    if pd.isna(final) or str(final).strip() in ('', 'Unknown', 'nan', 'NaN', 'None'):
        return 'Unknown'
    return final

final_result["Final Pack Size"] = final_result.apply(_choose_pack_size, axis=1)
final_result["Units"] = final_result["Units"].replace(unit_map)
final_result.to_excel(PS_MATCHED_OUTPUT, index=False)

# ================================
# Step 6: Amount words to numbers conversion
# ================================
print("\n=== Step 6: Converting amount in words to numbers ===")
df = ocr_output.copy()
spell = SpellChecker()
custom_words = ['lakh', 'lakhs', 'crore', 'crores', 'rupees', 'paise', 'only', 'inr', 'rs', 'lac', 'paise',
                'hundred', 'thousand', 'million', 'billion', 'trillion']
spell.word_frequency.load_words(custom_words)

number_dict = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000, 'lakh': 100000, 'lac': 100000,
    'crore': 10000000,
    'for': 4, 'fou': 4, 'our': 4, 'ive': 5, 'fiv': 5, 'iv': 4,
    'ight': 8, 'eigth': 8, 'aight': 8, 'to': 2, 'too': 2,
    'sax': 6, 'saxteen': 16, 'sinaty': 60, 'sinty': 60, 'sventy': 70,
    'sunding': 70, 'susting': 60, 'susty': 60, 'ninty': 90, 'ninaty': 90,
    'thite': 30, 'thirty': 30, 'lath': 100000, 'laith': 100000, 'lad': 100000,
    'thousnad': 1000, 'housand': 1000, 'tousand': 1000, 'thosand': 1000,
    'ixteen': 16, 'suteen': 16, 'sateen': 17, 'se': 6, 'tree': 3,
    'conty': 20, 'sainting': 60, 'tour': 4
}

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s\-]', ' ', text)
    return text

def replace_misspellings(text):
    misspellings = {
        'our': 'four', 'ive': 'five', 'ight': 'eight', 'laith': 'lakh',
        'lad': 'lakh', 'sax': 'six', 'ixteen': 'sixteen', 'sinty': 'sixty',
        'sinaty': 'sixty', 'sventy': 'seventy', 'sunding': 'seventy',
        'ninty': 'ninety', 'fity': 'fifty', 'thite': 'thirty', 'suteen': 'sixteen',
        'sateen': 'seventeen', 'for': 'four', 'to': 'two', 'tousand': 'thousand',
        'thosand': 'thousand', 'housand': 'thousand', 'lath': 'lakh', 'tree': 'three',
        'se': 'six', 'sainting': 'sixty', 'conty': 'twenty', 'susting': 'sixty',
        'susty': 'sixty', 'tour': 'four', 'saxteen': 'sixteen', 'light': 'eight',
        'singh': 'six', 'singhteen': 'sixteen'
    }
    tokens = text.split()
    corrected = [misspellings.get(w, w) for w in tokens]
    return ' '.join(corrected)

def correct_spelling(text):
    tokens = text.split()
    corrected = [spell.correction(w) if spell.correction(w) else w for w in tokens]
    return ' '.join(corrected)

def words_to_num(text):
    tokens = text.replace('-', ' ').split()
    total = 0
    current = 0
    for word in tokens:
        if word in ['and', 'only', 'rupees', 'rs', 'inr', 'paise', 'rupee']:
            continue
        if word in number_dict:
            scale = number_dict[word]
            if scale == 100:
                if current == 0:
                    current = 1
                current *= scale
            elif scale in [1000, 100000, 10000000]:
                if current == 0:
                    current = 1
                current *= scale
                total += current
                current = 0
            else:
                current += scale
    total += current
    return total

def process_amount(row):
    text = row['Total_Amount_in_words']
    text = normalize_text(text)
    text = replace_misspellings(text)
    text = correct_spelling(text)
    return words_to_num(text)

df['Total_Amount_in_words_converted'] = df.apply(process_amount, axis=1)
df.to_excel(os.path.join(POST_PROCESSING_DIR, 'output_highest_confidence_all_files_words_corrected.xlsx'), index=False)

# ================================
# Step 7: Parse numeric columns
# ================================
def parse_number(text):
    text = str(text).strip()
    if not text:
        return None
    text = text.split('\n')[0]
    text = re.sub(r'[^\d.,\s]', '', text)
    text = re.sub(r'(?<=\d) +(?=\d)', '.', text)
    text = text.replace(' ', '')
    match = re.search(r'([.,])(\d+)$', text)
    if match:
        decimal_sep = match.group(1)
        decimal_digits = match.group(2)
        integer_part = text[:match.start(1)]
        if decimal_sep == ',':
            integer_part = integer_part.replace('.', '')
            decimal_point = '.'
        else:
            integer_part = integer_part.replace(',', '')
            decimal_point = '.'
        integer_part = integer_part.replace(',', '').replace('.', '')
        number_str = integer_part + decimal_point + decimal_digits
    else:
        number_str = text.replace(',', '').replace('.', '')
    try:
        return float(number_str)
    except ValueError:
        return None

# columns_to_parse = ['total_amount', 'Total_Taxable_Amount', 'rate_per_unit', 'rate_per_ltr',
#                     'quantity', 'Taxable-Amount', 'Value', 'Volume', 'amount']
# for col in columns_to_parse:
#     df[f'{col}_in_Numbers'] = df[col].apply(lambda x: parse_number(str(x)))


columns_to_parse = ['total_amount', 'Total_Taxable_Amount', 'rate_per_unit', 'rate_per_ltr',
                    'quantity', 'Taxable-Amount', 'Value', 'Volume', 'amount']
for col in columns_to_parse:
    if col in df.columns:
        df[f'{col}_in_Numbers'] = df[col].apply(lambda x: parse_number(str(x)))
    else:
        print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")

output_file_numbers = os.path.join(POST_PROCESSING_DIR, 'output_highest_confidence_all_files_words_corrected_numbers_converted.xlsx')
df.to_excel(output_file_numbers, index=False)
print(f"Parsed numbers saved to {output_file_numbers}")

# ================================
# Step 8: Amount corrections and Green Code logic
# ================================
print("\n=== Step 7: Amount corrections and Green Code logic ===")
tolerance_percentage = 30
additional_tolerance = 20
df.rename(columns={
    'Taxable-Amount_in_Numbers': 'Taxable Amount',
    'Value_in_Numbers': 'Amt_w/o_disc_tax',
    'amount_in_Numbers': 'Amt',
    'rate_per_unit_in_Numbers': 'Price',
    'rate_per_ltr_in_Numbers': 'Rate_per_ltr',
    'quantity_in_Numbers': 'Qty',
    'Volume_in_Numbers': 'Vol'
}, inplace=True)
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(subset=['Taxable Amount', 'Amt_w/o_disc_tax', 'Amt'], how='all').reset_index(drop=True).copy()

def is_within_tolerance(base, target, tolerance):
    base_arr = np.asarray(base, dtype=float).reshape(-1)
    target_arr = np.asarray(target, dtype=float).reshape(-1)
    valid = ~(np.isnan(base_arr) | np.isnan(target_arr)) & (base_arr > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        deviation = np.abs((target_arr - base_arr) / base_arr) * 100
    result = np.zeros(len(base_arr), dtype=bool)
    result[valid] = deviation[valid] <= tolerance
    return pd.Series(result, index=base.index)

df['Amt_Correct'] = is_within_tolerance(df['Taxable Amount'], df['Amt'], tolerance_percentage) | is_within_tolerance(df['Amt_w/o_disc_tax'], df['Amt'], tolerance_percentage)

mask_missing_price = (df['Price'].isna()) & (df['Amt_Correct']) & df['Qty'].notna() & (df['Qty'] > 0)
df.loc[mask_missing_price, 'Price'] = df.loc[mask_missing_price, 'Amt'] / df.loc[mask_missing_price, 'Qty']
mask_missing_qty = (df['Qty'].isna()) & (df['Amt_Correct']) & df['Price'].notna() & (df['Price'] > 0)
df.loc[mask_missing_qty, 'Qty'] = df.loc[mask_missing_qty, 'Amt'] / df.loc[mask_missing_qty, 'Price']

df['Potential_Error_in'] = df.apply(lambda row: 'Price or Qty' if row['Amt_Correct'] else 'Amt, Taxable Amount, or Amt_w/o_disc_tax', axis=1)

def get_expected_amt(row):
    if pd.notna(row.get('Price')) and pd.notna(row.get('Qty')):
        return row['Price'] * row['Qty']
    if pd.notna(row.get('Rate_per_ltr')) and pd.notna(row.get('Vol')):
        return row['Rate_per_ltr'] * row['Vol']
    return np.nan

df['Expected_Amt'] = df.apply(get_expected_amt, axis=1)

def is_green_code(row):
    expected = row['Expected_Amt']
    if pd.notna(expected) and pd.notna(row['Amt']) and expected > 0:
        return abs(expected - row['Amt']) <= 0.30 * expected
    return False

df['Green_Code_Before'] = df.apply(is_green_code, axis=1)

def within_additional_tolerance(val1, val2, tolerance):
    if pd.notna(val1) and pd.notna(val2) and val1 > 0:
        return abs((val1 - val2) / val1) * 100 <= tolerance
    return False

def assign_amt(row):
    if not row['Green_Code_Before'] and pd.isna(row['Amt']):
        taxable = row['Taxable Amount']
        amt_wo = row['Amt_w/o_disc_tax']
        if pd.notna(taxable) and pd.notna(amt_wo) and within_additional_tolerance(taxable, amt_wo, additional_tolerance):
            return max(taxable, amt_wo)
    return row['Amt']

df['Amt'] = df.apply(assign_amt, axis=1)
df['Amt_Updated'] = df.apply(lambda row: 'Amt updated based on Taxable Amount and Amt_w/o_disc_tax' if (not row['Green_Code_Before'] and pd.isna(row['Amt']) and pd.notna(row['Taxable Amount']) and pd.notna(row['Amt_w/o_disc_tax']) and within_additional_tolerance(row['Taxable Amount'], row['Amt_w/o_disc_tax'], additional_tolerance)) else '', axis=1)
df['Green_Code_After'] = df.apply(is_green_code, axis=1)

output_amt_file = os.path.join(POST_PROCESSING_DIR, 'output_with_amt_and_green_code.xlsx')
df.to_excel(output_amt_file, index=False)

# ================================
# Step 9: Advanced correction logic
# ================================
tax_wo_disc_tolerance = 20
amt_deviation_tolerance = 30

def calc_dev(base, target):
    if pd.notna(base) and pd.notna(target) and base != 0:
        return abs((target - base) / base) * 100
    return np.nan

df['Deviation_Amt_Taxable'] = df.apply(lambda row: calc_dev(row['Taxable Amount'], row['Amt']), axis=1)
df['Deviation_Amt_Wo_Disc'] = df.apply(lambda row: calc_dev(row['Amt_w/o_disc_tax'], row['Amt']), axis=1)
df['Deviation_PriceQty_Taxable'] = df.apply(lambda row: calc_dev(row['Taxable Amount'], row['Expected_Amt']) if pd.notna(row['Expected_Amt']) else np.nan, axis=1)
df['Deviation_PriceQty_Amt_Wo_Disc'] = df.apply(lambda row: calc_dev(row['Amt_w/o_disc_tax'], row['Expected_Amt']) if pd.notna(row['Expected_Amt']) else np.nan, axis=1)

condition_deviation = (((df['Deviation_Amt_Taxable'] >= amt_deviation_tolerance) & df['Taxable Amount'].notna()) |
                       ((df['Deviation_Amt_Wo_Disc'] >= amt_deviation_tolerance) & df['Amt_w/o_disc_tax'].notna()))
condition_price_qty = (((df['Deviation_PriceQty_Taxable'] <= tax_wo_disc_tolerance) & df['Taxable Amount'].notna()) |
                       ((df['Deviation_PriceQty_Amt_Wo_Disc'] <= tax_wo_disc_tolerance) & df['Amt_w/o_disc_tax'].notna()))
combined_condition = condition_deviation & condition_price_qty
both_within_tolerance = ((df['Deviation_PriceQty_Taxable'] <= tax_wo_disc_tolerance) &
                         (df['Deviation_PriceQty_Amt_Wo_Disc'] <= tax_wo_disc_tolerance))
final_condition = combined_condition & ((~df['Taxable Amount'].isna() & ~df['Amt_w/o_disc_tax'].isna() & both_within_tolerance) |
                                        (df['Taxable Amount'].isna() | df['Amt_w/o_disc_tax'].isna()))

df['Amt_Corrected'] = df['Amt']
df.loc[final_condition, 'Amt_Corrected'] = df.loc[final_condition, ['Taxable Amount', 'Amt_w/o_disc_tax']].max(axis=1)

def recalc_green_code(row, tolerance=0.30):
    expected = row['Expected_Amt']
    if pd.notna(expected) and pd.notna(row['Amt_Corrected']) and expected > 0:
        return abs(expected - row['Amt_Corrected']) <= tolerance * expected
    return False

df['Green_Code_After_Corrected'] = df.apply(recalc_green_code, axis=1)
df['Amt_Corrected_Flag'] = np.where(final_condition, 'Amt corrected to higher of Taxable Amount and Amt_w/o_disc_tax', '')
df.drop(['Deviation_Amt_Taxable', 'Deviation_Amt_Wo_Disc', 'Deviation_PriceQty_Taxable', 'Deviation_PriceQty_Amt_Wo_Disc'], axis=1, inplace=True)

output_corrected_final = os.path.join(POST_PROCESSING_DIR, 'output_with_amt_and_green_code_output_corrected_final.xlsx')
df.to_excel(output_corrected_final, index=False)
print(f"Final corrections saved to {output_corrected_final}")

# ================================
# Step 10: Quantity corrections based on renditions
# ================================
print("\n=== Step 8: Quantity corrections ===")
def generate_substrings(number):
    num_str = str(int(number)) if not pd.isna(number) else ""
    return [num_str[i:j] for i in range(len(num_str)) for j in range(i+1, len(num_str)+1)]

df['quantity'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(int)
all_renditions = df['quantity'].apply(generate_substrings)
max_rend = max(all_renditions.apply(len)) if len(all_renditions) > 0 else 0
for i in range(max_rend):
    df[f'rendition {i+1}'] = all_renditions.apply(lambda x: x[i] if i < len(x) else None)

def find_correct_quantity(row):
    correct_qty = row['Qty']
    if pd.isna(row.get('Price')) or pd.isna(row.get('Amt')):
        return correct_qty
    for i in range(1, max_rend+1):
        col = f'rendition {i}'
        if col in row and row[col] is not None:
            try:
                rend = int(row[col])
                calc_amt = rend * float(row['Price'])
                lower = float(row['Amt']) * 0.75
                upper = float(row['Amt']) * 1.25
                if lower <= calc_amt <= upper:
                    return rend
            except:
                pass
    return correct_qty

df['correct quantity'] = df.apply(find_correct_quantity, axis=1)

# ================================
# Step 11: Decimal correction for price/amount
# ================================
def decimal_corrector(row):
    try:
        if pd.notna(row.get('Rate_per_ltr')) and pd.notna(row.get('Vol')):
            price = int(row['Rate_per_ltr'])
            qty = int(row['Vol'])
            amt = int(row['Amt_Corrected'])
        else:
            price = int(row['Price'])
            qty = int(row['correct quantity'])
            amt = int(row['Amt_Corrected'])
        price_str = str(price)
        if len(price_str) >= 2:
            rectified_price = float(price_str[:-2] + "." + price_str[-2:])
        else:
            rectified_price = float(price_str)
        price_calc = rectified_price * qty
        price_lb = amt * 0.75
        price_ub = amt * 1.25

        amt_str = str(amt)
        if len(amt_str) >= 2:
            rectified_amt = float(amt_str[:-2] + "." + amt_str[-2:])
        else:
            rectified_amt = float(amt_str)
        calc_amt = price * qty
        amt_lb = calc_amt * 0.75
        amt_ub = calc_amt * 1.25

        if price_lb <= price_calc <= price_ub:
            return (rectified_price, qty, amt)
        elif amt_lb <= rectified_amt <= amt_ub:
            return (price, qty, rectified_amt)
        else:
            return (price, qty, amt)
    except:
        return ("NA", "NA", "NA")

df[['Fin_Price', 'Fin_Qty', 'Fin_Amt']] = df.apply(decimal_corrector, axis=1, result_type='expand')
df.to_csv(CORRECTIONS_COMPLETE_CSV, index=False)

# ================================
# Step 12: Final SKU mapping with pack size
# ================================
print("\n=== Step 9: Final SKU mapping ===")
calc_df = pd.read_csv(CORRECTIONS_COMPLETE_CSV)
calc_df = calc_df[['S. No.', 'file_name', 'Fin_Price', 'Fin_Qty', 'Fin_Amt']]
calc_df['Fin_Price'] = calc_df['Fin_Price'].replace({0: "NA"})
calc_df['Fin_Qty'] = calc_df['Fin_Qty'].replace({0: "NA"})
calc_df['Fin_Amt'] = calc_df['Fin_Amt'].replace({0: "NA"})
calc_df = calc_df.replace({"NA": np.nan}).dropna()
calc_df['Metrics inline flag'] = calc_df.apply(lambda z: 1 if abs((z['Fin_Price'] * z['Fin_Qty']) - z['Fin_Amt']) / z['Fin_Amt'] < 0.3 else 0, axis=1)
calc_df = calc_df[calc_df['Metrics inline flag'] == 1]
calc_df['City'] = calc_df['file_name'].apply(lambda x: x.split('_')[2] if len(x.split('_')) > 2 else '')
calc_df['Outlet Name'] = calc_df['file_name'].apply(lambda x: x.split('_')[3] if len(x.split('_')) > 3 else '')
calc_df['Outlet Code'] = calc_df['file_name'].apply(lambda x: x.split('_')[4] if len(x.split('_')) > 4 else '')

sku_df = pd.read_csv(MASTER_MATCHING_CSV)[['S. No.', 'SKU Name']]
sku_master_df = master_file
sku_combined = sku_df.merge(sku_master_df[['SKU Name', 'Brand Name', 'Product']].drop_duplicates())
ps_matched = pd.read_excel(PS_MATCHED_OUTPUT)
sku_combined = sku_combined.merge(ps_matched[['S. No.', 'Final Pack Size', 'Final SKU Name']], on='S. No.', how='left')
sku_combined['Final Pack Size'] = sku_combined['Final Pack Size'].fillna('Unknown').replace({np.nan: 'Unknown'})
sku_combined['Final Pack Size'] = sku_combined['Final Pack Size'].apply(lambda x: 'Unknown' if pd.isna(x) or str(x).strip() in ('', 'nan', 'Unknown', 'None') else x)

combined_df = calc_df.merge(sku_combined, how='inner')
combined_df['Final SKU Name'] = combined_df.apply(
    lambda s: f"{s['SKU Name']}-{s['Final Pack Size']}" if str(s['Final Pack Size']) not in ('Unknown', 'nan', '') and pd.notna(s['Final Pack Size']) else str(s['SKU Name']),
    axis=1
)
combined_df.to_csv(INTERMEDIATE_FILE_CSV, index=False)
combined_df.drop(columns=['SKU Name'], inplace=True)
source_df = combined_df

# ================================
# Step 13: Handle unknown pack sizes using average price
# ================================
# print("\n=== Step 10: Handling unknown pack sizes ===")
# def process_string(s):
#     parts = s.split('-')
#     return '-'.join(parts) if len(parts) == 2 else '-'.join(parts[:-1])

# avg_price_df["SKU Name"] = avg_price_df["SKU"].apply(process_string)
# avg_price_df['Full Pack Size'] = avg_price_df["SKU"].apply(lambda x: x.split('-')[-1])
# avg_price_df = avg_price_df[~avg_price_df['Full Pack Size'].apply(lambda x: not bool(re.search(r'\d', x.split('-')[-1])))]
# avg_price_df['avg_price'] = avg_price_df.groupby('SKU Name')['Price'].transform('mean')
# avg_price_df = avg_price_df.drop_duplicates(subset=['SKU Name'], keep='first')

# subset_df = source_df[(source_df['Final Pack Size'] == 'Unknown') & (source_df['Brand Name'] != 'Other Brands')]
# rest_df = source_df[~source_df.index.isin(subset_df.index)]

# def convert_int(x):
#     if isinstance(x, float) and x.is_integer():
#         return int(x)
#     return x

# def find_closest_match(num, num_list):
#     return min(num_list, key=lambda x: abs(float(x) - float(num)))

# def find_packsize(row, size_df, avg_price_df):
#     row_df = pd.DataFrame([row], columns=subset_df.columns)
#     compare_df = row_df.merge(avg_price_df, how='inner', left_on='Final SKU Name', right_on='SKU Name')
#     if compare_df['avg_price'].empty:
#         return 'Unknown'
#     sku_name = str(compare_df['SKU Name'].iloc[0])
#     master_subset = size_df[size_df['SKU Name with packsize'].str.startswith(sku_name)]
#     size_options = master_subset['packsize_only'].dropna().astype(float).tolist()
#     if not size_options:
#         return 'Unknown'
#     fin_price = row['Fin_Price']
#     avg_price = float(compare_df['avg_price'].iloc[0])
#     est_packsize = fin_price / avg_price
#     matched = find_closest_match(est_packsize, size_options)
#     matched = convert_int(matched)
#     unit = master_subset['unit_only'].iloc[0] if not master_subset.empty else ''
#     return f"{matched} {unit}"

# subset_df['Final Pack Size'] = subset_df.apply(lambda x: find_packsize(x, size_df, avg_price_df), axis=1)
# subset_df['Final SKU Name'] = subset_df.apply(lambda x: f"{x['Final SKU Name']}-{x['Final Pack Size']}" if x['Final Pack Size'] != 'Unknown' else x['Final SKU Name'], axis=1)

# final_result = pd.concat([subset_df, rest_df], ignore_index=True)

# ================================
# Step 13: Handle unknown pack sizes using average price
# ================================
print("\n=== Step 10: Handling unknown pack sizes ===")

# Prepare average price data (always needed, but only used if subset exists)
def process_string(s):
    parts = s.split('-')
    return '-'.join(parts) if len(parts) == 2 else '-'.join(parts[:-1])

avg_price_df["SKU Name"] = avg_price_df["SKU"].apply(process_string)
avg_price_df['Full Pack Size'] = avg_price_df["SKU"].apply(lambda x: x.split('-')[-1])
avg_price_df = avg_price_df[~avg_price_df['Full Pack Size'].apply(lambda x: not bool(re.search(r'\d', x.split('-')[-1])))]
avg_price_df['avg_price'] = avg_price_df.groupby('SKU Name')['Price'].transform('mean')
avg_price_df = avg_price_df.drop_duplicates(subset=['SKU Name'], keep='first')

# Filter rows that need pack size resolution
subset_df = source_df[(source_df['Final Pack Size'] == 'Unknown') & (source_df['Brand Name'] != 'Other Brands')]

if not subset_df.empty:
    def convert_int(x):
        if isinstance(x, float) and x.is_integer():
            return int(x)
        return x

    def find_closest_match(num, num_list):
        return min(num_list, key=lambda x: abs(float(x) - float(num)))

    def find_packsize(row):
        row_df = pd.DataFrame([row], columns=subset_df.columns)
        compare_df = row_df.merge(avg_price_df, how='inner', left_on='Final SKU Name', right_on='SKU Name')
        if compare_df['avg_price'].empty:
            return 'Unknown'
        sku_name = str(compare_df['SKU Name'].iloc[0])
        master_subset = size_df[size_df['SKU Name with packsize'].str.startswith(sku_name)]
        size_options = master_subset['packsize_only'].dropna().astype(float).tolist()
        if not size_options:
            return 'Unknown'
        fin_price = row['Fin_Price']
        avg_price = float(compare_df['avg_price'].iloc[0])
        est_packsize = fin_price / avg_price
        matched = find_closest_match(est_packsize, size_options)
        matched = convert_int(matched)
        unit = master_subset['unit_only'].iloc[0] if not master_subset.empty else ''
        return f"{matched} {unit}"

    subset_df['Final Pack Size'] = subset_df.apply(lambda x: find_packsize(x), axis=1)
    subset_df['Final SKU Name'] = subset_df.apply(lambda x: f"{x['Final SKU Name']}-{x['Final Pack Size']}" if x['Final Pack Size'] != 'Unknown' else x['Final SKU Name'], axis=1)

    rest_df = source_df[~source_df.index.isin(subset_df.index)]
    final_result = pd.concat([subset_df, rest_df], ignore_index=True)
else:
    # No rows to process – use original source_df as is
    final_result = source_df.copy()
    print("No unknown pack sizes to handle.")


master_list = list(size_df['Final Name'])
final_result = final_result.merge(size_df[['SKU Name with packsize']], how='left', left_on='Final SKU Name', right_on='SKU Name with packsize')
nan_final = final_result[final_result['SKU Name with packsize'].isnull()]
nan_final = nan_final[nan_final['Final Pack Size'] != 'Unknown']
nan_final = nan_final[nan_final['Brand Name'] != 'Other Brands']
others_df = final_result[final_result['Brand Name'] == 'Other Brands']
others_df = others_df[others_df['SKU Name with packsize'].isnull()]
known_final = final_result[final_result['SKU Name with packsize'].notnull()]

nan_final['SKU Name'] = nan_final['Final SKU Name'].apply(process_string)

def convert_to_int(value):
    try:
        return int(float(value))
    except:
        return value

def check_packsize_match(row, master_list):
    orig_sku = str(row['SKU Name'])
    orig_pack = str(row['Final Pack Size'])
    candidate = f"{orig_sku}-{orig_pack}"
    if candidate in master_list:
        return candidate
    unit_only = re.sub(r'[^a-zA-Z]', '', orig_pack)
    packsize_only = re.sub(r'[^0-9.]', '', orig_pack)
    if packsize_only:
        pack_num = float(packsize_only)
        pack_num = pack_num * 1000 if pack_num < 1 else pack_num
        pack_num = convert_to_int(pack_num)
        unit_conv = 'ML' if unit_only == 'L' else unit_only
        candidate2 = f"{orig_sku}-{pack_num}{unit_conv}" if 'K' in unit_only else f"{orig_sku}-{pack_num} {unit_conv}"
        if candidate2 in master_list:
            return candidate2
        candidate3 = f"{orig_sku}-1" if 'Tools for Paints' in orig_sku else candidate2
        if candidate3 in master_list:
            return candidate3
        unit_title = unit_only.title()
        candidate4 = f"{orig_sku}-{pack_num} {unit_title}"
        if candidate4 in master_list:
            return candidate4
        unit_gm = 'GM' if 'Pidicrete URP' in orig_sku and pack_num < 100 else unit_title
        candidate5 = f"{orig_sku}-{pack_num} {unit_gm}"
        if 'Roff' in orig_sku or 'Nerolac - Tile Adhesives' in orig_sku or 'British Paints' in orig_sku:
            candidate5 = orig_sku
        if candidate5 in master_list:
            return candidate5
        candidate6 = f"{orig_sku}-{packsize_only}0 {unit_only}"
        if candidate6 in master_list:
            return candidate6
        candidate7 = f"{orig_sku}-{packsize_only}"
        if candidate7 in master_list:
            return candidate7
    return f"{orig_sku}- {orig_pack}"

nan_final['Last SKU Name'] = nan_final.apply(lambda x: check_packsize_match(x, master_list), axis=1)
known_final['Last SKU Name'] = known_final['Final SKU Name']

others_df['second part'] = others_df['Final SKU Name'].apply(lambda x: '-'.join(x.split('-')[1:]) if '-' in x else '')
others_df['Packsize'] = others_df['second part'].apply(lambda x: x.split('-')[1] if '-' in x else '')
others_df['Last SKU Name'] = others_df.apply(lambda x: f"{str(x['Brand Name']).strip()} - {str(x['Product']).strip()} - {x['Packsize']}" if x['Packsize'] else f"{str(x['Brand Name']).strip()} - {str(x['Product']).strip()}", axis=1)
others_df.drop(columns=['second part', 'Packsize'], inplace=True)

ps_matched_df = pd.concat([nan_final, known_final, others_df], ignore_index=True)
ps_matched_df.to_excel(REFER_DOC_XLSX, index=False)

# ================================
# Step 14: Final data file export
# ================================
data_file_df = ps_matched_df[['S. No.', 'City', 'Outlet Code', 'Outlet Name', 'Fin_Price', 'Fin_Qty', 'Fin_Amt', 'Brand Name', 'Product', 'Last SKU Name']]
data_file_df.to_excel(DATA_FILE_XLSX, index=False)

print("\n=== Processing complete ===")
print(f"Final data file saved to: {DATA_FILE_XLSX}")