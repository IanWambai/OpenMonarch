from pdfminer.high_level import extract_text
import os

# Define file paths
pdf_file = "dataset/history.pdf"  # Ensure this path is correct
output_txt_file = "dataset/african_history_clean.txt"

# Ensure output directory exists
os.makedirs("dataset", exist_ok=True)

# Extract text from PDF
text = extract_text(pdf_file)

# Save as a clean text file
with open(output_txt_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"PDF extraction complete. Text saved in {output_txt_file}")
