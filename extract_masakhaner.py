from datasets import load_dataset
import os

# List of all available language configs
languages = ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 
             'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul']

# Define output directory
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Define output file path
output_file = os.path.join(output_dir, "african_languages_clean.txt")

def extract_text(languages, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for lang in languages:
            print(f"Processing language: {lang}")
            dataset = load_dataset("masakhane/masakhaner2", lang)
            for split in dataset.keys():
                for example in dataset[split]:
                    text = " ".join(example["tokens"])  # Extract words only
                    f.write(text + "\n")

# Run extraction
extract_text(languages, output_file)

print(f"Extraction complete. Data saved in {output_file}")
