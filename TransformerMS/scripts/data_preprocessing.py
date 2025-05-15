import os
import re
import emoji
import logging

# BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ----------- Logging Configuration -----------
log_dir = os.path.join(BASE_DIR, 'Logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------- Regex Definitions -----------
english_regex = r"[a-zA-Z0-9\s]"                   # English letters, numbers, spaces
math_symbols  = r"[\+\-\*/=<>∑∫√πθΣ∂∞]"             # Common math symbols
special_chars = r"[\.,!?;:'\"()\[\]{}#@%^&*_~]"     # Common punctuation


# ----------- Utility Functions -----------

def load_file(file_path):
    """Load a text file and return its contents."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug("Loaded file: %s", file_path)
        return content
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Error loading file %s: %s", file_path, e)
        raise

def clean_text(text):
    """Clean text using defined regex rules and emoji support."""
    return "".join(
        c for c in text
        if re.match(english_regex, c) or
           re.match(math_symbols, c) or
           re.match(special_chars, c) or
           emoji.is_emoji(c)
    )

def save_cleaned_text(text, output_path):
    """Save cleaned text to the output file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.debug("Saved cleaned file to: %s", output_path)
    except Exception as e:
        logger.error("Error saving cleaned file to %s: %s", output_path, e)
        raise


# ----------- Main Function -----------
def main():
    try:
        raw_dir = os.path.join(BASE_DIR,"Data", "Raw")
        clean_dir = os.path.join(BASE_DIR,"Data", "Clean")
        os.makedirs(clean_dir, exist_ok=True)

        # File definitions
        datasets = {
            "L1_ChildrenStories.txt": "L1_cleaned.txt",
            "L2_BookCorpus.txt": "L2_cleaned.txt",
            "L3_CNN_DailyMail.txt": "L3_cleaned.txt",
            "L4_S2ORC.txt": "L4_cleaned.txt",
        }

        logger.info("Starting data cleaning...")

        for raw_file, clean_file in datasets.items():
            raw_path = os.path.join(raw_dir, raw_file)
            clean_path = os.path.join(clean_dir, clean_file)

            logger.info("Processing %s...", raw_file)
            text = load_file(raw_path)
            cleaned = clean_text(text)
            save_cleaned_text(cleaned, clean_path)

        logger.info("✅ All datasets cleaned and saved successfully!")

    except Exception as e:
        logger.error("❌ Data cleaning failed: %s", e)
        print(f"Error: {e}")

# ----------- Entry Point -----------
if __name__ == "__main__":
    main()
