import pandas as pd
import os

# === Step 1: Load Excel ===
excel_path = "pdb_data.xlsx"
column_name = "PDBID"  # change this to your actual column name

df = pd.read_excel(excel_path)
excel_ids = set(df[column_name].astype(str).str.strip().str.lower())

# === Step 2: Process text files ===
input_folder = "text_files"
output_folder = "filtered_files"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):
        input_path = os.path.join(input_folder, file_name)

        with open(input_path, "r") as f:
            lines = [line.strip().lower() for line in f if line.strip()]

        total_ids = len(lines)

        # === Step 3: Filter ===
        matched_ids = [pid for pid in lines if pid in excel_ids]
        matched_count = len(matched_ids)

        # === Step 4: Save ===
        output_path = os.path.join(
            output_folder,
            file_name.replace(".txt", "_filtered.txt")
        )

        with open(output_path, "w") as f:
            for pid in matched_ids:
                f.write(pid + "\n")

        # === Step 5: Print stats ===
        print(f"{file_name}: {matched_count} / {total_ids} IDs matched")