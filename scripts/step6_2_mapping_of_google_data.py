


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
from pathlib import Path
import argparse
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker

# Download NLTK data if needed
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
# Configuration
# ============================================================================
BASE_DIR = Path.cwd()
SKU_MAPPING_DIR = BASE_DIR / "SKU_mapping"
POST_PROCESSING_DIR = BASE_DIR / "post_processing files"

# Input files – will be overridden by command line argument
OCR_INPUT_CSV = BASE_DIR / "output_highest_confidence_all_files.csv"
SUPPLIER_RE_FILE = SKU_MAPPING_DIR / "Supplier_RE.xlsx"

# Mapping files
DICT_XLSX = SKU_MAPPING_DIR / "sku_mapping_dictionaries.xlsx"
KEYWORDS_XLSX = SKU_MAPPING_DIR / "Keywords.xlsx"
BRAND_DICT_XLSX = SKU_MAPPING_DIR / "product_brand_dict.xlsx"
COLOUR_MAPPING_XLSX = SKU_MAPPING_DIR / "colour_mapping_dict.xlsx"
SHORT_FORM_XLSX = SKU_MAPPING_DIR / "short_form_dictionary.xlsx"
AVG_PRICE_XLSX = BASE_DIR / "Paints CSV Report - May'25.xlsx"
SKU_MASTER_XLSX = SKU_MAPPING_DIR / "Paint - SKU Master - 06-01-2026.xlsx"

# Output files
RAW_SKU_EXCEL = BASE_DIR / "Rawsku.xlsx"
PROCESSED_DESC_EXCEL = SKU_MAPPING_DIR / "processed_product_descriptions.xlsx"
HIGH_FREQ_OUTPUT = SKU_MAPPING_DIR / "untagged_high_frequency_words.xlsx"
CORRECTED_FILE_CSV = SKU_MAPPING_DIR / "Corrected File.csv"
MASTER_MATCHING_CSV = SKU_MAPPING_DIR / "master_matching.csv"
PS_MATCHED_OUTPUT = SKU_MAPPING_DIR / "PS_matched_output.xlsx"
CORRECTIONS_COMPLETE_CSV = BASE_DIR / "Corrections complete.csv"
INTERMEDIATE_FILE_CSV = BASE_DIR / "Intermediate file.csv"
REFER_DOC_XLSX = BASE_DIR / "Refer Doc.xlsx"
DATA_FILE_XLSX = BASE_DIR / "Data_File_by_google.xlsx"

# Create directories
SKU_MAPPING_DIR.mkdir(exist_ok=True)
POST_PROCESSING_DIR.mkdir(exist_ok=True)

# ============================================================================
# Helper functions (unchanged)
# ============================================================================
def split_and_strip(input_string: str) -> str:
    """Remove the last part if it contains a digit (pack size)."""
    parts = input_string.split('-')
    if re.search(r'\d', parts[-1]):
        parts.pop()
    return '-'.join(parts)


def load_mapping_data():
    """Load all mapping Excel files and return dictionaries."""
    print("Loading mapping data...")
    # Sheets from sku_mapping_dictionaries.xlsx
    corrections_df = pd.read_excel(DICT_XLSX, sheet_name="corrections")
    high_freq_series_df = pd.read_excel(DICT_XLSX, sheet_name="high_freq_series_names")
    unit_map_df = pd.read_excel(DICT_XLSX, sheet_name="unit_map")
    unit_list_df = pd.read_excel(DICT_XLSX, sheet_name="unit_list")
    series_list_df = pd.read_excel(DICT_XLSX, sheet_name="series_list")
    brand_supplier_df = pd.read_excel(DICT_XLSX, sheet_name="brand_supplier_list")
    brand_var_df = pd.read_excel(DICT_XLSX, sheet_name="brands_var_list")

    # Keywords.xlsx
    product_keywords_df = pd.read_excel(KEYWORDS_XLSX, sheet_name="Product Final")
    brand_keywords_df = pd.read_excel(KEYWORDS_XLSX, sheet_name="Brand Final")
    not_paints_keyword = list(pd.read_excel(KEYWORDS_XLSX, sheet_name="not_paint")["Not Paint Keyword"])
    common_keywords_df = pd.read_excel(KEYWORDS_XLSX, sheet_name="common_keywords")
    common_keywords_list = common_keywords_df["Words"].tolist()

    # Other mappings
    brand_dict_df = pd.read_excel(BRAND_DICT_XLSX, sheet_name="Brand Name Mapping")
    color_list_df = pd.read_excel(COLOUR_MAPPING_XLSX, sheet_name="Colour List")
    color_mapping_df = pd.read_excel(COLOUR_MAPPING_XLSX, sheet_name="Colour Dictionary")
    product_brand_mapping_df = pd.read_excel(BRAND_DICT_XLSX, sheet_name="Product Brand Mapping")
    short_form_dict_df = pd.read_excel(SHORT_FORM_XLSX)

    # Convert to dictionaries / lists
    units_list = unit_list_df["Units"].tolist()
    unit_map = unit_map_df.set_index("Unit")["Standardized Unit"].to_dict()
    corrections = corrections_df.set_index("Expression")["Correction"].to_dict()
    product_keywords = product_keywords_df.groupby("Product")["Keyword"].unique().to_dict()
    brand_keywords = brand_keywords_df.groupby("Brand")["Keyword"].unique().to_dict()
    brand_dict = brand_dict_df.groupby("Brand")["Keyword"].unique().to_dict()
    color_list = color_list_df["Colours"].tolist()
    color_mapping = color_mapping_df.set_index("Short Form")["Colour"].to_dict()
    product_brand_mapping = product_brand_mapping_df.groupby("Product")["Brand"].unique().to_dict()
    short_form_dict = short_form_dict_df.set_index("Short Form")["Expansion"].to_dict()
    brand_supplier = brand_supplier_df["Brands"].tolist()
    brands_var_lst = brand_var_df["Brands"].tolist()
    series_list = series_list_df['Series'].tolist()

    # Reverse mapping product -> brands
    product_brand_reverse = {}
    for product, brands in product_brand_mapping.items():
        for brand in brands:
            product_brand_reverse.setdefault(product, []).append(brand)

    # Sort units for pack size regex
    units_sorted = sorted(units_list, key=len, reverse=True)
    pack_size_units_pattern = r'(?:^|(?<=[\s\-_]))[\(]*\s*(\d+(?:\.\d+)?(?:\s*[xX*]\s*\d+(?:\.\d+)?)?)(?:\s*[-/_]?\s*)(' + '|'.join(re.escape(u) for u in units_sorted) + r')\b'

    return {
        'corrections': corrections,
        'product_keywords': product_keywords,
        'brand_keywords': brand_keywords,
        'brand_dict': brand_dict,
        'color_list': color_list,
        'color_mapping': color_mapping,
        'product_brand_reverse': product_brand_reverse,
        'short_form_dict': short_form_dict,
        'brand_supplier': brand_supplier,
        'brands_var_lst': brands_var_lst,
        'series_list': series_list,
        'not_paints_keyword': not_paints_keyword,
        'common_keywords_list': common_keywords_list,
        'unit_map': unit_map,
        'pack_size_units_pattern': pack_size_units_pattern,
        'units_sorted': units_sorted,
    }


def parse_product_description(description, mappings):
    """Parse raw product description into structured fields."""
    brand = 'Unknown'
    series_name = 'Unknown'
    product_name = 'Unknown'
    color = 'Unknown'
    pack_size = 'Unknown'
    units = 'Unknown'

    if pd.isna(description):
        description = ''
    elif not isinstance(description, str):
        description = str(description)

    description = description.lower()
    tokens = word_tokenize(description)
    tokens_series = pd.Series(tokens).replace(mappings['short_form_dict'])
    description = " ".join(tokens_series.tolist())

    desc = description.upper()
    desc = re.sub(r'[^\x00-\x7F]+', ' ', desc)

    # Apply corrections
    for pattern, replacement in mappings['corrections'].items():
        desc = re.sub(pattern, replacement, desc)
    desc = re.sub(r'\s+', ' ', desc).strip()

    # Map color abbreviations
    for abbr, full_name in mappings['color_mapping'].items():
        desc = re.sub(r'\b' + re.escape(abbr) + r'\b', full_name, desc)

    # Extract pack size (last match)
    pack_matches = list(re.finditer(mappings['pack_size_units_pattern'], desc))
    if pack_matches:
        pm = pack_matches[-1]
        pack_size = pm.group(1).replace(' ', '')
        units = pm.group(2)
        desc = desc.replace(pm.group(0), '').strip()

    # Extract color
    for col in mappings['color_list']:
        if re.search(r'\b' + re.escape(col.upper()) + r'\b', desc):
            color = col
            if col != 'WHITE':
                desc = re.sub(r'\b' + re.escape(col.upper()) + r'\b', '', desc).strip()
            break

    key_word_list = mappings['common_keywords_list']

    # Extract series and brand
    series_found = False
    for brand_key, series_list in mappings['brand_keywords'].items():
        for series in series_list:
            if re.search(r'\b' + re.escape(series.upper()) + r'\b', desc, re.IGNORECASE):
                series_name = series
                brand = brand_key
                if series.lower() not in key_word_list:
                    desc = re.sub(r'\b' + re.escape(series.upper()) + r'\b', '', desc).strip()
                series_found = True
                break
        if series_found:
            break

    # Special Birla case
    excluded = ['Opus', 'OPUS', 'opus', 'White', 'white', 'WHITE', 'HIL', 'Hil', 'hil']
    pattern_birla = rf'\bBirla\b(?!\s+({"|".join(excluded)}))'
    if re.search(pattern_birla, desc, re.IGNORECASE):
        brand = "Birla HIL_Birla Opus_Birla White"

    # If brand still unknown, use brand_dict
    if brand == 'Unknown':
        brand_found = False
        for full_name, abbrs in mappings['brand_dict'].items():
            for abbr in abbrs:
                if re.search(r'\b' + re.escape(abbr.upper()) + r'\b', desc, re.IGNORECASE):
                    brand = full_name
                    if brand.lower() not in key_word_list:
                        desc = re.sub(r'\b' + re.escape(abbr.upper()) + r'\b', '', desc).strip()
                    brand_found = True
                    break
            if brand_found:
                break

    # Extract product name
    for prod, keywords in mappings['product_keywords'].items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw.upper()) + r'\b', desc, re.IGNORECASE):
                product_name = prod
                desc = re.sub(r'\b' + re.escape(kw.upper()) + r'\b', '', desc, flags=re.IGNORECASE).strip()
                break
        if product_name != 'Unknown':
            break

    # Assign brand from product if needed
    if brand == 'Unknown' and product_name != 'Unknown':
        possible = mappings['product_brand_reverse'].get(product_name, [])
        if len(possible) == 1:
            brand = possible[0]
        elif len(possible) > 1:
            match = process.extractOne(desc, possible, scorer=fuzz.token_sort_ratio)
            if match and match[1] > 85:
                brand = match[0]

    remaining = ' '.join(desc.split())
    return {
        'Brand': brand,
        'Series Name': series_name,
        'Product': product_name,
        'Color': color,
        'Pack Size': pack_size,
        'Units': units,
        'Remaining Description': remaining
    }



def extract_high_frequency_words(descriptions, tagged_words_set, freq_threshold=10):
    """Extract high-frequency words not in tagged set."""
    # Convert all descriptions to strings and replace NaN with empty string
    descriptions = [str(d) if pd.notna(d) else '' for d in descriptions]
    combined = ' '.join(descriptions).upper()
    combined = re.sub(r'[^\x00-\x7F]+', ' ', combined)
    combined = re.sub(r'[^\w\s]', ' ', combined)
    combined = re.sub(r'\d+', ' ', combined)
    words = combined.split()
    counts = Counter(words)
    high_freq = {w: c for w, c in counts.items() if c >= freq_threshold}
    untagged = {w: c for w, c in high_freq.items() if w not in tagged_words_set}
    df_out = pd.DataFrame(list(untagged.items()), columns=['Word', 'Frequency'])
    return df_out.sort_values('Frequency', ascending=False)

def collect_tagged_words(mappings):
    """Collect all words from parsing dictionaries."""
    tagged = set()
    # corrections
    for pattern in mappings['corrections'].keys():
        word = re.sub(r'[\\\^\$\.\|\?\*\+\(\)\[\]\{\}]', '', pattern)
        word = word.replace('\\b', '').replace('\\', '').strip()
        if word:
            tagged.add(word.upper())
    # brand_dict
    for brand, abbrs in mappings['brand_dict'].items():
        tagged.add(brand.upper())
        for abbr in abbrs:
            tagged.add(abbr.upper())
    # product_keywords
    for prod, keywords in mappings['product_keywords'].items():
        tagged.add(prod.upper())
        for kw in keywords:
            tagged.add(kw.upper())
    # colors
    for col in mappings['color_list']:
        tagged.add(col.upper())
    for abbr, full in mappings['color_mapping'].items():
        tagged.add(abbr.upper())
        tagged.add(full.upper())
    # brand_keywords series
    for brand, series_list in mappings['brand_keywords'].items():
        for series in series_list:
            tagged.add(series.upper())
    return tagged


def prepare_master_skus():
    """Load and preprocess master SKU file."""
    print("Preparing master SKU data...")
    size_df = pd.read_excel(SKU_MASTER_XLSX, sheet_name='FInal SKu')
    size_df = size_df.drop(columns=['Period', 'Comment', 'Price Segment'], errors='ignore')

    master_file = size_df.drop(columns='Base Name', errors='ignore')
    master_file = master_file.rename(columns={
        "Droid SKU New Name Dec'25": 'SKU Name',
        "Company Name": 'Brand Name',
        "Brand Name": 'Series Name',
        "Category - MP": 'Product'
    })
    master_file['SKU Name'] = master_file["SKU Name"].apply(split_and_strip)
    master_file['Series Name'] = master_file['Series Name'].str.upper()
    master_file['Brand Name'] = master_file['Brand Name'].replace({'Asian Paints': 'asianpaints'})
    # British Paints override
    mask_british = master_file['SKU Name'].str.contains('British Paints', na=False)
    master_file.loc[mask_british, 'Brand Name'] = 'British Paints'

    # Prepare size_df for later use
    size_df = size_df[["Droid SKU New Name Dec'25", "Pack Size", "Company Name"]]
    size_df = size_df[size_df['Company Name'] != 'Other Brands']
    size_df = size_df.rename(columns={"Droid SKU New Name Dec'25": 'SKU Name with packsize'})
    size_df['SKU Name'] = size_df['SKU Name with packsize'].apply(
        lambda x: split_and_strip(x) if 'Tools for Paints' not in x else x
    )
    size_df['Final Name'] = size_df['SKU Name with packsize']
    size_df['Pack Size'] = size_df['Pack Size'].astype(str)
    size_df['unit_only'] = size_df['Pack Size'].str.replace(r'[^a-zA-Z]', '', regex=True)
    size_df['packsize_only'] = size_df['Pack Size'].str.extract(r'(\d*\.?\d+)')

    return master_file, size_df


def parse_number(text):
    """Convert string with possible thousand/decimal separators to float."""
    if not text or pd.isna(text):
        return None
    text = str(text).strip().split('\n')[0]
    text = re.sub(r'[^\d.,\s]', '', text)
    text = re.sub(r'(?<=\d) +(?=\d)', '.', text)
    text = text.replace(' ', '')
    match = re.search(r'([.,])(\d+)$', text)
    if match:
        sep = match.group(1)
        digits = match.group(2)
        integer_part = text[:match.start(1)]
        if sep == ',':
            integer_part = integer_part.replace('.', '')
            decimal_point = '.'
        else:
            integer_part = integer_part.replace(',', '')
            decimal_point = '.'
        integer_part = integer_part.replace(',', '').replace('.', '')
        num_str = integer_part + decimal_point + digits
    else:
        num_str = text.replace(',', '').replace('.', '')
    try:
        return float(num_str)
    except ValueError:
        return None


def words_to_number(text, number_dict):
    """Convert English words to number."""
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


def process_amount_text(text, spell, number_dict):
    """Full pipeline: normalize, correct spelling, convert to number."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s\-]', ' ', text)
    # Common misspellings
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
    corrected = [misspellings.get(t, t) for t in tokens]
    text = ' '.join(corrected)
    # Spell check
    tokens = text.split()
    corrected_tokens = [spell.correction(w) if spell.correction(w) else w for w in tokens]
    text = ' '.join(corrected_tokens)
    return words_to_number(text, number_dict)


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='SKU Mapping and Invoice Processing')
    parser.add_argument('--ocr_csv', required=True, help='Path to OCR input CSV file')
    args = parser.parse_args()

    global OCR_INPUT_CSV
    OCR_INPUT_CSV = Path(args.ocr_csv)

    print("=" * 60)
    print("SKU Mapping and Invoice Processing Pipeline")
    print("=" * 60)

    # Step 1: Load mappings
    mappings = load_mapping_data()

    # Step 2: Load OCR data and parse descriptions
    print("Loading OCR data...")
    ocr_df = pd.read_csv(OCR_INPUT_CSV)


    if 'S. No.' not in ocr_df.columns:
        ocr_df.insert(0, 'S. No.', [f'G{i+1}' for i in range(len(ocr_df))])
        print("Added 'S. No.' column with sequential numbers (G1, G2, G3...).")
    
    # Ensure description column is string (avoid float issues)
    if 'description' in ocr_df.columns:
        ocr_df['description'] = ocr_df['description'].astype(str).fillna('')


    input_df = ocr_df[['S. No.', 'description', 'supplier_name']].rename(columns={'description': 'Description'})
    input_df.to_excel(RAW_SKU_EXCEL, index=False)

    print("Parsing product descriptions...")
    tagged_words = collect_tagged_words(mappings)
    untagged_df = extract_high_frequency_words(input_df['Description'], tagged_words)
    untagged_df.to_excel(HIGH_FREQ_OUTPUT, index=False)

    parsed_results = input_df['Description'].apply(lambda d: parse_product_description(d, mappings))
    parsed_df = pd.DataFrame(parsed_results.tolist())
    result_df = pd.concat([input_df, parsed_df], axis=1)
    result_df.to_excel(PROCESSED_DESC_EXCEL, index=False)

    # Step 3: Supplier-based brand correction
    print("Supplier-based brand correction...")
    df = result_df.copy()
    known_df = df[df['Brand'] != "Unknown"]
    unknown_df = df[df['Brand'] == "Unknown"]

    re_df = pd.read_excel(SUPPLIER_RE_FILE)
    re_dict = re_df.groupby('Supplier Name RE')['Brand'].unique().to_dict()

    def get_brand(supplier):
        if not isinstance(supplier, str):
            return supplier
        for key in re_dict:
            if key in supplier.lower():
                brands = re_dict[key]
                if len(brands) > 1:
                    return "_".join(brands[:3])
                else:
                    return brands[0]
        return supplier

    unknown_df['supplier_name_clean'] = unknown_df['supplier_name'].apply(get_brand)
    known_df['supplier_name_clean'] = known_df['supplier_name'].apply(get_brand)

    # British Paints override
    for df_temp in [known_df, unknown_df]:
        mask = df_temp['supplier_name'].str.lower().str.contains('british paints division', na=False)
        df_temp.loc[mask, 'supplier_name_clean'] = 'British Paints'

    unknown_df['Brand'] = unknown_df.apply(
        lambda z: z['supplier_name_clean'] if z['supplier_name_clean'] in mappings['brand_supplier'] else 'Unknown', axis=1
    )

    df = pd.concat([known_df, unknown_df], ignore_index=True)

    # Split multi-brand entries
    for multi_brand, split_name in [
        ("Birla HIL_Birla Opus_Birla White", "Birla HIL_Birla Opus_Birla White"),
        ("JK Cement_JK MAXX Paints_JK Lakshmi", "JK Cement_JK MAXX Paints_JK Lakshmi")
    ]:
        rem = df[df['Brand'] != multi_brand]
        split = df[df['Brand'] == multi_brand]
        parts = []
        for rpt in range(3):
            temp = split.copy()
            temp['Brand'] = split_name.split("_")[rpt]
            parts.append(temp)
        df = pd.concat([rem] + parts, ignore_index=True)

    # Flag non-paint products
    def product_check(pr_name):
        if isinstance(pr_name, str):
            for kw in mappings['not_paints_keyword']:
                if kw in pr_name.lower():
                    return 1
        return 0

    df['Not Paint Product Flag'] = df['Description'].apply(product_check)
    paints_df = df[df['Not Paint Product Flag'] == 0]
    paints_df['Brand'] = paints_df['Brand'].replace({"Unknown": "Other Brands", "OTHER BRANDS": "Other Brands"})
    all_brands_df = pd.concat([paints_df, known_df], ignore_index=True)

    brand_supplier_refined = list(set(mappings['brand_supplier']) - {"Birla HIL_Birla Opus_Birla White", 'JK Cement_JK MAXX Paints_JK Lakshmi'})
    all_brands_df['Brand'] = all_brands_df.apply(
        lambda z: z['supplier_name_clean'] if (z['supplier_name_clean'] != z['Brand'] and
                                                z['supplier_name_clean'] in brand_supplier_refined and
                                                z['Brand'] != 'British Paints') else z['Brand'], axis=1
    )
    all_brands_df.to_csv(CORRECTED_FILE_CSV, index=False)

    df = all_brands_df.drop_duplicates().dropna(subset=['Description'])
    df["Brand"] = df["Brand"].str.title()
    brand_replace = {
        "Asian Paints": "asianpaints", "Jsw": "JSW", "Jk Cement": "JK Cement",
        "Asianpaints": "asianpaints", "Dr. Fixit": "DR. FIXIT", 'Birla Hil': "Birla HIL",
        "Jk Maxx Paints": "JK MAXX Paints", 'Jk Cement ': 'JK Cement', 'Mrf': 'MRF'
    }
    df['Brand'] = df['Brand'].replace(brand_replace)

    # Step 4: Prepare master database for fuzzy matching
    master_file, size_df = prepare_master_skus()
    db_df = master_file.dropna(subset=['SKU Name']).copy()

    def sku_refiner(row):
        sku = row['SKU Name']
        brand = row['Brand Name']
        cleaned = sku.replace("-", "").replace(brand, "").strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.replace("(cw tinting base)", "")
        return cleaned

    db_df['SKU refined'] = db_df.apply(sku_refiner, axis=1)
    db_df = db_df[['Brand Name', 'SKU Name', 'SKU refined', 'Product', 'Series Name']].drop_duplicates()
    db_df.rename(columns={"Brand Name": "Brand"}, inplace=True)

    # Clean description column
    df['Desc_cleaned'] = df['Description'].str.lower()
    df['Desc_cleaned'] = df['Desc_cleaned'].str.replace(r'\d', ' ', regex=True)
    df['Desc_cleaned'] = df['Desc_cleaned'].str.replace(r'[^a-z\s]', ' ', regex=True)
    df['Desc_cleaned'] = df['Desc_cleaned'].str.replace(r'\s+', ' ', regex=True)

    stop_words = set(stopwords.words('english')).union(
        {'ml', 'ltr', 'lt', 'kg', 'l', "buy", "for", "price", "litre", "at", "from",
         "india", "best", "prices", "online", "get", 'gst', 'hi', 'cgst', 'bag', 'gms',
         'central', 'state', 'ap'}
    )

    def cleaner(txt):
        tokens = word_tokenize(txt)
        tokens = [t for t in tokens if t not in stop_words]
        tokens_series = pd.Series(tokens).replace(mappings['short_form_dict'])
        tokens = [t for t in tokens_series if len(t) >= 2]
        cleaned = " ".join(tokens)
        for b in mappings['brands_var_lst']:
            if cleaned.startswith(b):
                cleaned = re.sub(b, "", cleaned, count=1)
        return cleaned.strip()

    df['Desc_cleaned'] = df['Desc_cleaned'].apply(cleaner)
    df = df[df['Desc_cleaned'] != ""]

    # Step 5: Fuzzy matching to SKU refined
    print("Fuzzy matching descriptions to SKU refined...")
    def matcher_brand_specific(row):
        string, brand, product, series = row['Desc_cleaned'], row['Brand'], row['Product'], row['Series Name']
        if brand != "Unknown":
            sub_df = db_df[db_df['Brand'] == brand]
            if series in mappings['series_list']:
                sub_df = sub_df[sub_df['Series Name'].str.contains(series, na=False)]
            if product != "Unknown":
                if product == "Emulsion":
                    sub_df = sub_df[sub_df['Product'].isin(["Interior Emulsion", "Exterior Emulsion"])]
                elif product == "Putty":
                    sub_df = sub_df[sub_df['Product'].isin(["Cement Putty", "Acrylic Putty"])]
                else:
                    sub_df = sub_df[sub_df['Product'] == product]
        else:
            sub_df = db_df
            if product != "Unknown":
                if product == "Emulsion":
                    sub_df = sub_df[sub_df['Product'].isin(["Interior Emulsion", "Exterior Emulsion"])]
                elif product == "Putty":
                    sub_df = sub_df[sub_df['Product'].isin(["Cement Putty", "Acrylic Putty"])]
                else:
                    sub_df = sub_df[sub_df['Product'] == product]
        choices = sub_df['SKU refined'].unique()
        if len(choices) == 0:
            return (-1, -1)
        match = process.extractOne(string, choices, scorer=fuzz.token_sort_ratio)
        return (match[0], match[1]) if match else (-1, -1)

    df[['SKU refined', 'Match Score']] = df.apply(matcher_brand_specific, axis=1, result_type='expand')
    db_df_unique = db_df.drop_duplicates(subset=['SKU Name'])
    df_merged = pd.merge(df, db_df_unique[['SKU refined', 'SKU Name', 'Brand']], on=['SKU refined', 'Brand'], how='left')
    df_merged = df_merged.sort_values("Match Score", ascending=False).drop_duplicates(subset=['S. No.'], keep='first')
    df_merged['Brand'] = df_merged['Brand'].replace({'British Paints': "Berger"})
    df_merged.to_csv(MASTER_MATCHING_CSV, index=False)

    # Step 6: Pack size normalization and matching
    print("Pack size normalization and matching...")
    def parse_pack_size(pack_size):
        if pd.isna(pack_size) or not isinstance(pack_size, str):
            return None, 'Unknown'
        parts = pack_size.strip().split()
        if len(parts) == 2:
            try:
                return float(parts[0]), parts[1]
            except:
                return None, parts[1]
        else:
            m = re.match(r"^(\d+(\.\d+)?)(.*)$", pack_size.strip())
            if m:
                return float(m.group(1)), m.group(3).strip() or 'Unknown'
        return None, 'Unknown'

    def normalize_units(val, unit):
        if pd.isna(val) or pd.isna(unit):
            return None, 'Unknown'
        unit = unit.strip().upper()
        unit = mappings['unit_map'].get(unit, unit)
        try:
            val = float(val)
        except:
            return None, 'Unknown'
        if val is None or unit == 'Unknown':
            return None, 'Unknown'
        if unit == 'ML':
            val /= 1000.0
            unit = 'L'
        elif unit == 'G':
            val /= 1000.0
            unit = 'KG'
        return val, unit

    source_df = pd.read_csv(MASTER_MATCHING_CSV)
    master_df = master_file.copy()
    master_df[['Raw_Pack_Num', 'Raw_Unit']] = master_df['Pack Size'].apply(lambda x: pd.Series(parse_pack_size(x)))
    master_df[['Pack Size Num', 'Norm Units']] = master_df.apply(lambda r: pd.Series(normalize_units(r['Raw_Pack_Num'], r['Raw_Unit'])), axis=1)
    master_df['SKU_lower'] = master_df['SKU Name'].str.lower().str.strip()
    source_df[['Pack Size Num', 'Norm Units']] = source_df.apply(lambda r: pd.Series(normalize_units(r['Pack Size'], r['Units'])), axis=1)
    source_df['SKU_lower'] = source_df['SKU Name'].str.lower().str.strip()
    master_df['Pack_Full'] = master_df.apply(lambda r: f"{r['Pack Size Num']:.6g} {r['Norm Units']}" if r['Pack Size Num'] is not None and r['Norm Units'] != 'Unknown' else "Unknown", axis=1)
    source_df['Pack_Full'] = source_df.apply(lambda r: f"{r['Pack Size Num']:.6g} {r['Norm Units']}" if r['Pack Size Num'] is not None and r['Norm Units'] != 'Unknown' else "Unknown", axis=1)

    def fuzzy_match_sku_pack(row, master_df):
        src_sku = row['SKU_lower']
        pack_num = row['Pack Size Num']
        all_skus = master_df['SKU_lower'].unique()
        if len(all_skus) == 0 or not isinstance(src_sku, str):
            return None, None
        result = process.extractOne(src_sku, all_skus, scorer=fuzz.WRatio)
        if not result or result[1] < 60:
            return None, None
        best_sku = result[0]
        candidates = master_df[master_df['SKU_lower'] == best_sku].copy()
        if candidates.empty:
            return None, None
        if pack_num is not None and not candidates['Pack Size Num'].isna().all():
            candidates = candidates.dropna(subset=['Pack Size Num'])
            if not candidates.empty:
                candidates['diff'] = (candidates['Pack Size Num'] - pack_num).abs()
                best = candidates.loc[candidates['diff'].idxmin()]
                return best['SKU Name'], best['Pack_Full']
        best = candidates.iloc[0]
        return best['SKU Name'], None

    merged = pd.merge(source_df, master_df, left_on=['SKU_lower', 'Pack_Full'], right_on=['SKU_lower', 'Pack_Full'], how='left', suffixes=('', '_master'))
    merged['Matched SKU'] = None
    merged['Matched Pack_Full'] = None
    no_match = merged[merged['SKU Name_master'].isna()].copy()
    if not no_match.empty:
        no_match[['Matched SKU', 'Matched Pack_Full']] = no_match.apply(lambda r: pd.Series(fuzzy_match_sku_pack(r, master_df)), axis=1)
        merged.update(no_match[['Matched SKU', 'Matched Pack_Full']])
    merged['Final SKU Name'] = merged['Matched SKU'].fillna(merged['SKU Name_master'])
    merged['Final Pack Size'] = merged['Matched Pack_Full'].fillna(merged['Pack_Full'])
    final_result = merged[['S. No.', 'SKU Name', 'Pack Size', 'Units', 'Final SKU Name', 'Final Pack Size']]

    def choose_pack_size(row):
        orig = row['Pack Size']
        if pd.isna(orig) or str(orig).strip() in ('', 'Unknown', 'nan', 'NaN', 'None'):
            return 'Unknown'
        final = row['Final Pack Size']
        if pd.isna(final) or str(final).strip() in ('', 'Unknown', 'nan', 'NaN', 'None'):
            return 'Unknown'
        return final

    final_result["Final Pack Size"] = final_result.apply(choose_pack_size, axis=1)
    final_result["Units"] = final_result["Units"].replace(mappings['unit_map'])
    final_result.to_excel(PS_MATCHED_OUTPUT, index=False)

    # Step 7: Amount words to numbers
    print("Converting amount words to numbers...")
    spell = SpellChecker()
    custom_words = ['lakh', 'lakhs', 'crore', 'crores', 'rupees', 'paise', 'only', 'inr', 'rs', 'lac',
                    'hundred', 'thousand', 'million', 'billion', 'trillion']
    spell.word_frequency.load_words(custom_words)

    number_dict = {
        'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
        'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,
        'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,
        'eighty':80,'ninety':90,'hundred':100,'thousand':1000,'lakh':100000,'lac':100000,'crore':10000000,
        'for':4,'fou':4,'our':4,'ive':5,'fiv':5,'iv':4,'ight':8,'eigth':8,'aight':8,'to':2,'too':2,
        'sax':6,'saxteen':16,'sinaty':60,'sinty':60,'sventy':70,'sunding':70,'susting':60,'susty':60,
        'ninty':90,'ninaty':90,'thite':30,'thirty':30,'lath':100000,'laith':100000,'lad':100000,
        'thousnad':1000,'housand':1000,'tousand':1000,'thosand':1000,'ixteen':16,'suteen':16,
        'sateen':17,'se':6,'tree':3,'conty':20,'sainting':60,'tour':4
    }

    ocr_df['Total_Amount_in_words_converted'] = ocr_df['Total_Amount_in_words'].apply(
        lambda x: process_amount_text(x, spell, number_dict)
    )
    ocr_df.to_excel(POST_PROCESSING_DIR / "output_highest_confidence_all_files_words_corrected.xlsx", index=False)

    # Step 8: Parse numeric columns
    print("Parsing numeric columns...")
    num_cols = ['total_amount', 'Total_Taxable_Amount', 'unit_price', 'quantity',
                'Taxable-Amount', 'Value', 'Volume', 'amount']
    for col in num_cols:
        if col in ocr_df.columns:
            ocr_df[f'{col}_in_Numbers'] = ocr_df[col].apply(lambda x: parse_number(x))
    ocr_df.to_excel(POST_PROCESSING_DIR / "output_highest_confidence_all_files_words_corrected_numbers_converted.xlsx", index=False)

    # Step 9: Amount corrections and Green Code
    print("Amount corrections and Green Code...")
    df = ocr_df.copy()
    df.rename(columns={
        'Taxable-Amount_in_Numbers': 'Taxable Amount',
        'Value_in_Numbers': 'Amt_w/o_disc_tax',
        'amount_in_Numbers': 'Amt',
        'unit_price_in_Numbers': 'Price',
        'quantity_in_Numbers': 'Qty'
    }, inplace=True)
    df = df.dropna(subset=['Taxable Amount', 'Amt_w/o_disc_tax', 'Amt'], how='all').reset_index(drop=True)

    def within_tol(base, target, tol):
        if pd.notna(base) and pd.notna(target) and base > 0:
            return abs((target - base) / base) * 100 <= tol
        return False

    df['Amt_Correct'] = df.apply(lambda r: within_tol(r['Taxable Amount'], r['Amt'], 30) or within_tol(r['Amt_w/o_disc_tax'], r['Amt'], 30), axis=1)
    df['Potential_Error_in'] = df.apply(lambda r: 'Price or Qty' if r['Amt_Correct'] else 'Amt, Taxable Amount, or Amt_w/o_disc_tax', axis=1)

    def green_code(row):
        if pd.notna(row['Price']) and pd.notna(row['Qty']) and pd.notna(row['Amt']) and row['Price'] > 0 and row['Qty'] > 0:
            return abs((row['Price'] * row['Qty']) - row['Amt']) <= 0.30 * (row['Price'] * row['Qty'])
        return False

    df['Green_Code_Before'] = df.apply(green_code, axis=1)

    def assign_amt(row):
        if not row['Green_Code_Before'] and pd.isna(row['Amt']):
            taxable = row['Taxable Amount']
            amt_wo = row['Amt_w/o_disc_tax']
            if pd.notna(taxable) and pd.notna(amt_wo) and within_tol(taxable, amt_wo, 20):
                return max(taxable, amt_wo)
        return row['Amt']

    df['Amt'] = df.apply(assign_amt, axis=1)
    df['Amt_Updated'] = df.apply(lambda r: 'Amt updated' if (not r['Green_Code_Before'] and pd.isna(r['Amt']) and
                                                              pd.notna(r['Taxable Amount']) and pd.notna(r['Amt_w/o_disc_tax']) and
                                                              within_tol(r['Taxable Amount'], r['Amt_w/o_disc_tax'], 20)) else '', axis=1)
    df['Green_Code_After'] = df.apply(green_code, axis=1)
    df.to_excel(POST_PROCESSING_DIR / "output_with_amt_and_green_code.xlsx", index=False)

    # Step 10: Advanced correction logic
    print("Advanced correction logic...")
    def calc_dev(base, target):
        if pd.notna(base) and pd.notna(target) and base != 0:
            return abs((target - base) / base) * 100
        return np.nan

    df['Dev_Amt_Taxable'] = df.apply(lambda r: calc_dev(r['Taxable Amount'], r['Amt']), axis=1)
    df['Dev_Amt_WoDisc'] = df.apply(lambda r: calc_dev(r['Amt_w/o_disc_tax'], r['Amt']), axis=1)
    df['Dev_PQ_Taxable'] = df.apply(lambda r: calc_dev(r['Taxable Amount'], r['Price'] * r['Qty']) if pd.notna(r['Price']) and pd.notna(r['Qty']) else np.nan, axis=1)
    df['Dev_PQ_WoDisc'] = df.apply(lambda r: calc_dev(r['Amt_w/o_disc_tax'], r['Price'] * r['Qty']) if pd.notna(r['Price']) and pd.notna(r['Qty']) else np.nan, axis=1)

    cond_dev = ((df['Dev_Amt_Taxable'] >= 30) & df['Taxable Amount'].notna()) | ((df['Dev_Amt_WoDisc'] >= 30) & df['Amt_w/o_disc_tax'].notna())
    cond_pq = ((df['Dev_PQ_Taxable'] <= 20) & df['Taxable Amount'].notna()) | ((df['Dev_PQ_WoDisc'] <= 20) & df['Amt_w/o_disc_tax'].notna())
    both_tol = (df['Dev_PQ_Taxable'] <= 20) & (df['Dev_PQ_WoDisc'] <= 20)
    final_cond = cond_dev & cond_pq & ((~df['Taxable Amount'].isna() & ~df['Amt_w/o_disc_tax'].isna() & both_tol) | (df['Taxable Amount'].isna() | df['Amt_w/o_disc_tax'].isna()))

    df['Amt_Corrected'] = df['Amt']
    df.loc[final_cond, 'Amt_Corrected'] = df.loc[final_cond, ['Taxable Amount', 'Amt_w/o_disc_tax']].max(axis=1)

    def recalc_green(row, tol=0.30):
        if pd.notna(row['Price']) and pd.notna(row['Qty']) and pd.notna(row['Amt_Corrected']) and row['Price'] > 0 and row['Qty'] > 0:
            expected = row['Price'] * row['Qty']
            return abs(expected - row['Amt_Corrected']) <= tol * expected
        return False

    df['Green_Code_After_Corrected'] = df.apply(recalc_green, axis=1)
    df['Amt_Corrected_Flag'] = np.where(final_cond, 'Amt corrected to higher', '')
    df.drop(['Dev_Amt_Taxable', 'Dev_Amt_WoDisc', 'Dev_PQ_Taxable', 'Dev_PQ_WoDisc'], axis=1, inplace=True)
    df.to_excel(POST_PROCESSING_DIR / "output_with_amt_and_green_code_output_corrected_final.xlsx", index=False)

    # Step 11: Quantity correction using renditions and bounding box (if available)
    print("Quantity correction...")
    def gen_substrings(num):
        s = str(int(num)) if not pd.isna(num) else ""
        return [s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)]

    df['quantity'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(int)
    all_rend = df['quantity'].apply(gen_substrings)
    max_rend = max(all_rend.apply(len)) if len(all_rend) > 0 else 0
    for i in range(max_rend):
        df[f'rendition_{i+1}'] = all_rend.apply(lambda x: x[i] if i < len(x) else None)

    def find_correct_qty(row):
        if pd.isna(row.get('Price')) or pd.isna(row.get('Amt')):
            return row['Qty']
        for i in range(1, max_rend+1):
            col = f'rendition_{i}'
            if col in row and row[col] is not None:
                try:
                    rend = int(row[col])
                    calc = rend * float(row['Price'])
                    low = float(row['Amt']) * 0.75
                    high = float(row['Amt']) * 1.25
                    if low <= calc <= high:
                        return rend
                except:
                    pass
        return row['Qty']

    df['correct quantity'] = df.apply(find_correct_qty, axis=1)

    # Bounding box width comparison if column exists
    if 'quantity_bounding_box' in df.columns:
        def extract_x1_x2(box):
            if pd.isna(box):
                return None, None
            coords = re.findall(r"\(([^)]+)\)", str(box))
            if len(coords) < 2:
                return None, None
            p1 = tuple(map(float, coords[0].split(',')))
            p2 = tuple(map(float, coords[1].split(',')))
            return p1[0], p2[0]
        df[['x1', 'x2']] = df['quantity_bounding_box'].apply(lambda b: pd.Series(extract_x1_x2(b)))
        df['width'] = df['x2'] - df['x1']
        for fname in df['file_name'].unique():
            true_rows = df[(df['file_name'] == fname) & (df['Green_Code_After_Corrected'] == True)]
            if not true_rows.empty:
                true_width = true_rows['width'].iloc[0]
                mask = (df['file_name'] == fname) & (df['Green_Code_After_Corrected'] == False)
                df.loc[mask, 'quantity_box_wider'] = df.loc[mask, 'width'] > true_width
        df['quantity_box_wider'] = df['quantity_box_wider'].fillna(False)

    # Step 12: Decimal correction
    def decimal_corrector(row):
        try:
            price = int(row['Price'])
            qty = int(row['correct quantity'])
            amt = int(row['Amt_Corrected'])
            price_str = str(price)
            if len(price_str) >= 2:
                rect_price = float(price_str[:-2] + "." + price_str[-2:])
            else:
                rect_price = float(price_str)
            price_calc = rect_price * qty
            price_low = amt * 0.75
            price_high = amt * 1.25
            amt_str = str(amt)
            if len(amt_str) >= 2:
                rect_amt = float(amt_str[:-2] + "." + amt_str[-2:])
            else:
                rect_amt = float(amt_str)
            calc_amt = price * qty
            amt_low = calc_amt * 0.75
            amt_high = calc_amt * 1.25
            if price_low <= price_calc <= price_high:
                return (rect_price, qty, amt)
            elif amt_low <= rect_amt <= amt_high:
                return (price, qty, rect_amt)
            else:
                return (price, qty, amt)
        except:
            return ("NA", "NA", "NA")

    df[['Fin_Price', 'Fin_Qty', 'Fin_Amt']] = df.apply(decimal_corrector, axis=1, result_type='expand')
    df.to_csv(CORRECTIONS_COMPLETE_CSV, index=False)

    # Step 13: Final SKU mapping with pack size
    print("Final SKU mapping...")
    calc_df = pd.read_csv(CORRECTIONS_COMPLETE_CSV)
    calc_df = calc_df[['S. No.', 'file_name', 'Fin_Price', 'Fin_Qty', 'Fin_Amt']]
    calc_df['Fin_Price'] = calc_df['Fin_Price'].replace({0: "NA"})
    calc_df['Fin_Qty'] = calc_df['Fin_Qty'].replace({0: "NA"})
    calc_df['Fin_Amt'] = calc_df['Fin_Amt'].replace({0: "NA"})
    calc_df = calc_df.replace({"NA": np.nan}).dropna()
    calc_df['Metrics inline flag'] = calc_df.apply(lambda z: 1 if abs((z['Fin_Price'] * z['Fin_Qty']) - z['Fin_Amt']) / z['Fin_Amt'] < 0.3 else 0, axis=1)
    calc_df = calc_df[calc_df['Metrics inline flag'] == 1]
    # Extract metadata from file_name
    parts = calc_df['file_name'].str.split('_', expand=True)
    calc_df['City'] = parts[3] if parts.shape[1] > 3 else ''
    calc_df['Outlet Name'] = parts[4] if parts.shape[1] > 4 else ''
    calc_df['Outlet Code'] = parts[5] if parts.shape[1] > 5 else ''

    sku_df = pd.read_csv(MASTER_MATCHING_CSV)[['S. No.', 'SKU Name']]
    sku_master_df = master_file
    sku_combined = sku_df.merge(sku_master_df[['SKU Name', 'Brand Name', 'Product']].drop_duplicates())
    ps_matched = pd.read_excel(PS_MATCHED_OUTPUT)
    sku_combined = sku_combined.merge(ps_matched[['S. No.', 'Final Pack Size', 'Final SKU Name']], on='S. No.', how='left')
    sku_combined['Final Pack Size'] = sku_combined['Final Pack Size'].fillna('Unknown').replace({np.nan: 'Unknown'})
    sku_combined['Final Pack Size'] = sku_combined['Final Pack Size'].apply(lambda x: 'Unknown' if pd.isna(x) or str(x).strip() in ('', 'nan', 'Unknown', 'None') else x)

    combined_df = calc_df.merge(sku_combined, how='inner')
    combined_df['Final SKU Name'] = combined_df.apply(
        lambda s: f"{s['SKU Name']}-{s['Final Pack Size']}" if s['Final Pack Size'] not in ('Unknown', 'nan', '') and pd.notna(s['Final Pack Size']) else str(s['SKU Name']),
        axis=1
    )
    combined_df.to_csv(INTERMEDIATE_FILE_CSV, index=False)
    combined_df.drop(columns=['SKU Name'], inplace=True)

    # Step 14: Handle unknown pack sizes using average price
    print("Handling unknown pack sizes...")
    def process_string(s):
        parts = s.split('-')
        return '-'.join(parts) if len(parts) == 2 else '-'.join(parts[:-1])

    avg_price = pd.read_excel(AVG_PRICE_XLSX, sheet_name='Sheet2')
    avg_price["SKU Name"] = avg_price["SKU"].apply(process_string)
    avg_price['Full Pack Size'] = avg_price["SKU"].apply(lambda x: x.split('-')[-1])
    avg_price = avg_price[~avg_price['Full Pack Size'].apply(lambda x: not bool(re.search(r'\d', x.split('-')[-1])))]
    avg_price['avg_price'] = avg_price.groupby('SKU Name')['Price'].transform('mean')
    avg_price = avg_price.drop_duplicates(subset=['SKU Name'], keep='first')


    def find_closest_match(num, num_list):
        return min(num_list, key=lambda x: abs(float(x) - float(num)))

    def find_packsize(row):
        compare = pd.DataFrame([row]).merge(avg_price, how='inner', left_on='Final SKU Name', right_on='SKU Name')
        if compare['avg_price'].empty:
            return 'Unknown'
        sku_name = str(compare['SKU Name'].iloc[0])
        master_sub = size_df[size_df['SKU Name with packsize'].str.startswith(sku_name)]
        size_opts = master_sub['packsize_only'].dropna().astype(float).tolist()
        if not size_opts:
            return 'Unknown'
        est = row['Fin_Price'] / float(compare['avg_price'].iloc[0])
        match = find_closest_match(est, size_opts)
        if isinstance(match, float) and match.is_integer():
            match = int(match)
        unit = master_sub['unit_only'].iloc[0] if not master_sub.empty else ''
        return f"{match} {unit}"
    
    
    subset = combined_df[(combined_df['Final Pack Size'] == 'Unknown') & (combined_df['Brand Name'] != 'Other Brands')]
    rest = combined_df[~combined_df.index.isin(subset.index)]

    if not subset.empty:
    # Process only if subset is not empty
        subset['Final Pack Size'] = subset.apply(lambda x: find_packsize(x, size_df, avg_price), axis=1)
        subset['Final SKU Name'] = subset.apply(
            lambda x: f"{x['Final SKU Name']}-{x['Final Pack Size']}" if x['Final Pack Size'] != 'Unknown' else x['Final SKU Name'],
            axis=1
        )
        final_result = pd.concat([subset, rest], ignore_index=True)
    else:
        # If no rows to process, final_result is just rest (or combined_df if rest also empty)
        final_result = rest.copy() if not rest.empty else pd.DataFrame()
        print("No rows with unknown pack size found. Skipping pack size resolution.")

    # subset['Final Pack Size'] = subset.apply(find_packsize, axis=1)
    # subset['Final SKU Name'] = subset.apply(lambda x: f"{x['Final SKU Name']}-{x['Final Pack Size']}" if x['Final Pack Size'] != 'Unknown' else x['Final SKU Name'], axis=1)

    # final_result = pd.concat([subset, rest], ignore_index=True)
    master_list = list(size_df['Final Name'])
    final_result = final_result.merge(size_df[['SKU Name with packsize']], how='left', left_on='Final SKU Name', right_on='SKU Name with packsize')
    nan_final = final_result[final_result['SKU Name with packsize'].isnull()]
    nan_final = nan_final[nan_final['Final Pack Size'] != 'Unknown']
    nan_final = nan_final[nan_final['Brand Name'] != 'Other Brands']
    others = final_result[final_result['Brand Name'] == 'Other Brands']
    others = others[others['SKU Name with packsize'].isnull()]
    known_final = final_result[final_result['SKU Name with packsize'].notnull()]

    nan_final['SKU Name'] = nan_final['Final SKU Name'].apply(process_string)

    def convert_to_int(val):
        try:
            return int(float(val))
        except:
            return val

    def check_packsize_match(row):
        orig_sku = str(row['SKU Name'])
        orig_pack = str(row['Final Pack Size'])
        cand1 = f"{orig_sku}-{orig_pack}"
        if cand1 in master_list:
            return cand1
        unit_only = re.sub(r'[^a-zA-Z]', '', orig_pack)
        pack_only = re.sub(r'[^0-9.]', '', orig_pack)
        if pack_only:
            pack_num = float(pack_only)
            pack_num = pack_num * 1000 if pack_num < 1 else pack_num
            pack_num = convert_to_int(pack_num)
            unit_conv = 'ML' if unit_only == 'L' else unit_only
            cand2 = f"{orig_sku}-{pack_num}{unit_conv}" if 'K' in unit_only else f"{orig_sku}-{pack_num} {unit_conv}"
            if cand2 in master_list:
                return cand2
            cand3 = f"{orig_sku}-1" if 'Tools for Paints' in orig_sku else cand2
            if cand3 in master_list:
                return cand3
            unit_title = unit_only.title()
            cand4 = f"{orig_sku}-{pack_num} {unit_title}"
            if cand4 in master_list:
                return cand4
            unit_gm = 'GM' if 'Pidicrete URP' in orig_sku and pack_num < 100 else unit_title
            cand5 = f"{orig_sku}-{pack_num} {unit_gm}"
            if 'Roff' in orig_sku or 'Nerolac - Tile Adhesives' in orig_sku or 'British Paints' in orig_sku:
                cand5 = orig_sku
            if cand5 in master_list:
                return cand5
            cand6 = f"{orig_sku}-{pack_only}0 {unit_only}"
            if cand6 in master_list:
                return cand6
            cand7 = f"{orig_sku}-{pack_only}"
            if cand7 in master_list:
                return cand7
        return f"{orig_sku}- {orig_pack}"

    nan_final['Last SKU Name'] = nan_final.apply(check_packsize_match, axis=1)
    known_final['Last SKU Name'] = known_final['Final SKU Name']
    others['second part'] = others['Final SKU Name'].apply(lambda x: '-'.join(x.split('-')[1:]) if '-' in x else '')
    others['Packsize'] = others['second part'].apply(lambda x: x.split('-')[1] if '-' in x else '')
    others['Last SKU Name'] = others.apply(lambda x: f"{str(x['Brand Name']).strip()} - {str(x['Product']).strip()} - {x['Packsize']}" if x['Packsize'] else f"{str(x['Brand Name']).strip()} - {str(x['Product']).strip()}", axis=1)
    others.drop(columns=['second part', 'Packsize'], inplace=True)

    ps_matched_df = pd.concat([nan_final, known_final, others], ignore_index=True)
    ps_matched_df.to_excel(REFER_DOC_XLSX, index=False)

    # Final export
    data_file = ps_matched_df[['S. No.', 'City', 'Outlet Code', 'Outlet Name', 'Fin_Price', 'Fin_Qty', 'Fin_Amt', 'Brand Name', 'Product', 'Last SKU Name']]
    
  
    data_file.to_excel(DATA_FILE_XLSX, index=False)

    print("\n" + "=" * 60)
    print("Processing completed successfully!")
    print(f"Final data file: {DATA_FILE_XLSX}")
    print("=" * 60)


if __name__ == "__main__":
    main()