
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ============================================================================
# ALL PATHS ARE DEFINED HERE – CENTRAL CONTROL
# ============================================================================

# Step 4_1 (Birla White)
BIRLA_SCRIPT = r"D:\Yash\tag_using_llm\P&P Streamline\scripts\step4_1_invoice_data_extraction_birla_white.py"
BIRLA_INPUT = r"D:\Yash\tag_using_llm\P&P Streamline\invoices_for_all\birla_white"
BIRLA_OUTPUT = r"D:\Yash\tag_using_llm\P&P Streamline\invoices_for_all\birla_white\output_json"
BIRLA_API_KEY = "k_e5464b64cc97.3vmxz4fSnLXNOdSDF-yP8M9Fq0ocESZFjKeoGm58xbaA4aztCZu-IA"

# Step 4_2 (Other Brands)
OTHER_SCRIPT = r"D:\Yash\tag_using_llm\P&P Streamline\scripts\step4_2_invoice_data_extraction_other_brands.py"
OTHER_INPUT = r"D:\Yash\tag_using_llm\P&P Streamline\invoices_for_all\other_brand"
OTHER_OUTPUT = r"D:\Yash\tag_using_llm\P&P Streamline\invoices_for_all\other_brand\output_json"
OTHER_API_KEY = "k_e5464b64cc97.3vmxz4fSnLXNOdSDF-yP8M9Fq0ocESZFjKeoGm58xbaA4aztCZu-IA"

# Step 4_3 (Google Document AI OCR)
GOOGLE_SCRIPT = r"D:\Yash\tag_using_llm\P&P Streamline\scripts\step4_3_invoice_data_extraction_using_google.py"
GOOGLE_INPUT = r"D:\Yash\tag_using_llm\P&P Streamline\invoices_for_all\google"
OCR_CSV = r"D:\Yash\tag_using_llm\P&P Streamline\output_highest_confidence_all_files.csv"
SERVICE_ACCOUNT_JSON = r"D:\Yash\tag_using_llm\P&P Streamline\invoice-scanner-429009-0055d5568275.json"

# Step 5_1 (JSON to CSV mapping)
MAPPING_SCRIPT = r"D:\Yash\tag_using_llm\P&P Streamline\scripts\step5_1_llm_generated_json_column_mapping.py"
BIRLA_MAPPING_FILE = r"D:\Yash\tag_using_llm\P&P Streamline\column_mapping_file\final_column_mapping.xlsx"
OTHER_MAPPING_FILE = r"D:\Yash\tag_using_llm\P&P Streamline\column_mapping_file\column_mapping_birla_white.csv"
BIRLA_CSV_OUT = r"D:\Yash\tag_using_llm\P&P Streamline\line_items_birla_white.csv"
OTHER_CSV_OUT = r"D:\Yash\tag_using_llm\P&P Streamline\line_items_other_brands.csv"

# Step 6_1 (LLM SKU mapping)
STEP6_1_SCRIPT = r"D:\Yash\tag_using_llm\P&P Streamline\scripts\step6_1_mapping_of_llm_data.py"
MERGED_CSV = r"D:\Yash\tag_using_llm\P&P Streamline\merged_llm_line_items.csv"
LLM_OUTPUT_XLSX = r"D:\Yash\tag_using_llm\P&P Streamline\Data_File_by_llm.xlsx"

# Step 6_2 (Google SKU mapping)
STEP6_2_SCRIPT = r"D:\Yash\tag_using_llm\P&P Streamline\scripts\step6_2_mapping_of_google_data.py"
GOOGLE_OUTPUT_XLSX = r"D:\Yash\tag_using_llm\P&P Streamline\Data_File_by_google.xlsx"

# Final combined output
FINAL_CSV = r"D:\Yash\tag_using_llm\P&P Streamline\data_file.csv"

# Main log directory
MAIN_LOG_DIR = r"D:\Yash\tag_using_llm\P&P Streamline\main_logs"

# ============================================================================
# Setup logging
# ============================================================================
def setup_main_logging():
    os.makedirs(MAIN_LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(MAIN_LOG_DIR, f"main_orchestrator_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Main orchestrator log: {log_file}")
    return log_file

def run_script(cmd, name):
    """Run a command as subprocess, log output."""
    logging.info(f"Starting {name} with: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None, check=False)
        if result.stdout:
            logging.info(f"{name} STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"{name} STDERR:\n{result.stderr}")
        if result.returncode == 0:
            logging.info(f"{name} finished successfully.")
            return {"name": name, "status": "success", "returncode": 0}
        else:
            logging.error(f"{name} failed with exit code {result.returncode}.")
            return {"name": name, "status": "failure", "returncode": result.returncode}
    except Exception as e:
        logging.exception(f"Exception running {name}: {e}")
        return {"name": name, "status": "error", "error": str(e)}


    
def combine_and_rename_csv(file1, file2, output_file, rename_map={'value': 'Value'}):
    """Combine two CSV files, rename columns as per rename_map, and add S. No. column."""
    try:
        if not os.path.exists(file1):
            logging.warning(f"File not found: {file1}")
            return False
        if not os.path.exists(file2):
            logging.warning(f"File not found: {file2}")
            return False
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        combined = pd.concat([df1, df2], ignore_index=True)
        
        # Rename columns if they exist
        for old_name, new_name in rename_map.items():
            if old_name in combined.columns:
                combined.rename(columns={old_name: new_name}, inplace=True)
                logging.info(f"Renamed column '{old_name}' to '{new_name}'")
        
        # Add S. No. column with L1, L2, L3... sequence
        combined.insert(0, 'S. No.', [f'L{i+1}' for i in range(len(combined))])
        logging.info(f"Added 'S. No.' column with L1, L2, L3... sequence")
        
        combined.to_csv(output_file, index=False)
        logging.info(f"Combined CSV saved to {output_file} with {len(combined)} rows")
        return True
    except Exception as e:
        logging.error(f"Failed to combine CSV files: {e}")
        return False

def combine_excel_files(excel1, excel2, output_csv):
    """Read two Excel files, concatenate rows, save as CSV."""
    try:
        if not os.path.exists(excel1):
            logging.warning(f"Excel file not found: {excel1}")
            return False
        if not os.path.exists(excel2):
            logging.warning(f"Excel file not found: {excel2}")
            return False
        df1 = pd.read_excel(excel1)
        df2 = pd.read_excel(excel2)
        combined = pd.concat([df1, df2], ignore_index=True)
        combined.to_csv(output_csv, index=False)
        logging.info(f"Final combined CSV saved to {output_csv} with {len(combined)} rows")
        return True
    except Exception as e:
        logging.error(f"Failed to combine Excel files: {e}")
        return False

# ============================================================================
# Main orchestration
# ============================================================================
def main():
    setup_main_logging()
    logging.info("=" * 80)
    logging.info("Starting parallel execution of step4_1, step4_2, step4_3")
    logging.info("=" * 80)

    # Define commands for the three extraction steps
    cmd_birla = [sys.executable, BIRLA_SCRIPT,
                 "--input_folder", BIRLA_INPUT,
                 "--output_folder", BIRLA_OUTPUT,
                 "--api_key", BIRLA_API_KEY]

    cmd_other = [sys.executable, OTHER_SCRIPT,
                 "--input_folder", OTHER_INPUT,
                 "--output_folder", OTHER_OUTPUT,
                 "--api_key", OTHER_API_KEY]

    cmd_google = [sys.executable, GOOGLE_SCRIPT,
                  "--input_folder", GOOGLE_INPUT,
                  "--output_csv", OCR_CSV,
                  "--service_account_json", SERVICE_ACCOUNT_JSON]

    scripts = [
        (cmd_birla, "Birla_White_Extraction"),
        (cmd_other, "Other_Brands_Extraction"),
        (cmd_google, "Google_DocAI_OCR")
    ]

    # Verify script files exist
    for cmd, name in scripts:
        if not os.path.isfile(cmd[1]):
            logging.error(f"Script not found: {cmd[1]} for {name}")
            sys.exit(1)
        logging.info(f"Found script: {name} -> {cmd[1]}")

    # Run extraction steps in parallel
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {executor.submit(run_script, cmd, name): name for cmd, name in scripts}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                logging.error(f"{name} generated exception: {exc}")
                results.append({"name": name, "status": "unexpected_error", "error": str(exc)})

    # Summary of extraction phase
    logging.info("=" * 80)
    logging.info("EXTRACTION PHASE COMPLETED")
    for res in results:
        logging.info(f"  {res['name']}: {res.get('status', 'unknown').upper()}")
    logging.info("=" * 80)

    # Check success of Birla, Other, and Google steps
    birla_status = next((r for r in results if r['name'] == "Birla_White_Extraction"), None)
    other_status = next((r for r in results if r['name'] == "Other_Brands_Extraction"), None)
    google_status = next((r for r in results if r['name'] == "Google_DocAI_OCR"), None)

    # ========================================================================
    # LLM path: Birla + Other -> step5_1 -> combine -> step6_1
    # ========================================================================
    if birla_status and birla_status.get('status') == 'success':
        logging.info("Birla White extraction succeeded. Running JSON to CSV mapping...")
        cmd_map_birla = [sys.executable, MAPPING_SCRIPT,
                         BIRLA_OUTPUT, BIRLA_CSV_OUT, BIRLA_MAPPING_FILE]
        map_res = run_script(cmd_map_birla, "Step5_1_Birla_Mapping")
        logging.info(f"Birla mapping result: {map_res.get('status')}")
    else:
        logging.warning("Birla White extraction failed, skipping its mapping.")

    if other_status and other_status.get('status') == 'success':
        logging.info("Other Brands extraction succeeded. Running JSON to CSV mapping...")
        cmd_map_other = [sys.executable, MAPPING_SCRIPT,
                         OTHER_OUTPUT, OTHER_CSV_OUT, OTHER_MAPPING_FILE]
        map_res = run_script(cmd_map_other, "Step5_1_Other_Mapping")
        logging.info(f"Other brands mapping result: {map_res.get('status')}")
    else:
        logging.warning("Other Brands extraction failed, skipping its mapping.")

    # Combine Birla and Other CSV files into merged_llm_line_items.csv
    if (birla_status and birla_status.get('status') == 'success' and
        other_status and other_status.get('status') == 'success'):
        logging.info("Both extractions succeeded. Combining CSV files into merged file...")
        combine_success = combine_and_rename_csv(BIRLA_CSV_OUT, OTHER_CSV_OUT, MERGED_CSV)
        if combine_success:
            logging.info(f"Merged CSV ready: {MERGED_CSV}")
            logging.info("Running step6_1 SKU mapping on merged data...")
            cmd_step6_1 = [sys.executable, STEP6_1_SCRIPT, "--input_csv", MERGED_CSV]
            step6_1_res = run_script(cmd_step6_1, "Step6_1_LLM_SKU_Mapping")
            logging.info(f"Step6_1 result: {step6_1_res.get('status')}")
        else:
            logging.error("Failed to combine CSV files. Skipping step6_1.")
    else:
        logging.warning("One or both extraction steps failed. Cannot combine CSV files. Skipping step6_1.")

    # ========================================================================
    # Google path: step4_3 -> step6_2
    # ========================================================================
    if google_status and google_status.get('status') == 'success':
        logging.info("Google OCR succeeded. Running step6_2 SKU mapping on OCR data...")
        cmd_step6_2 = [sys.executable, STEP6_2_SCRIPT, "--ocr_csv", OCR_CSV]
        step6_2_res = run_script(cmd_step6_2, "Step6_2_Google_SKU_Mapping")
        logging.info(f"Step6_2 result: {step6_2_res.get('status')}")
    else:
        logging.warning("Google OCR failed. Skipping step6_2.")

    # ========================================================================
    # Final combination of Data_File_by_llm.xlsx and Data_File_by_google.xlsx
    # ========================================================================
    llm_status = step6_1_res if 'step6_1_res' in locals() else None
    google_map_status = step6_2_res if 'step6_2_res' in locals() else None

    if llm_status and llm_status.get('status') == 'success' and google_map_status and google_map_status.get('status') == 'success':
        logging.info("Both LLM and Google SKU mapping succeeded. Combining final Excel files...")
        combine_excel_files(GOOGLE_OUTPUT_XLSX, LLM_OUTPUT_XLSX, FINAL_CSV)
    else:
        logging.warning("One of the SKU mapping steps failed. Skipping final combination.")

    logging.info("=" * 80)
    logging.info("ALL PROCESSING COMPLETED")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()