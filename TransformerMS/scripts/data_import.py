import os
import datasets
import logging
from dotenv import load_dotenv
import yaml

# BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
# Load environment variables
load_dotenv(override=True)

# Setup logging

log_dir = os.path.join(BASE_DIR, 'Logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_import")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "data_import.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# File paths
raw_dir = os.path.join(BASE_DIR, 'Data','Raw')
# raw_dir = "Data/Raw"
os.makedirs(raw_dir, exist_ok=True)
# L1_FILE = "Data/Raw/L1_ChildrenStories.txt"
L1_FILE = os.path.join(raw_dir, 'L1_ChildrenStories.txt')
# L2_FILE = "Data/Raw/L2_BookCorpus.txt"
L2_FILE = os.path.join(raw_dir, 'L2_BookCorpus.txt')
# L3_FILE = "Data/Raw/L3_CNN_DailyMail.txt"
L3_FILE = os.path.join(raw_dir, 'L3_CNN_DailyMail.txt')
# L4_FILE = "Data/Raw/L4_S2ORC.txt"
L4_FILE = os.path.join(raw_dir, 'L4_S2ORC.txt')


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load parameters: %s", e)
        raise


def process_dataset(dataset_name,file_size, output_file, text_keys, dataset_config=None):
    try:
        logger.info("Processing dataset: %s", dataset_name)

        if dataset_config:
            dataset = datasets.load_dataset(dataset_name, dataset_config, split="train")
        else:
            dataset = datasets.load_dataset(dataset_name, split="train")

        logger.debug("Loaded %s with %d records", dataset_name, len(dataset))
        logger.debug("Sample keys: %s", list(dataset[0].keys()))

        missing_keys = [key for key in text_keys if key not in dataset[0]]
        if missing_keys:
            logger.error("Missing keys in dataset %s: %s", dataset_name, missing_keys)
            return

        with open(output_file, "w", encoding="utf-8") as f:
            size = 0
            for item in dataset:
                text = " ".join([item[key] for key in text_keys if key in item])
                f.write(text + "\n")
                size += len(text.encode("utf-8"))
                # if size >= 50 * 1024 * 1024:  # 100MB limit
                if size >= file_size * 1024 * 1024:  # 100MB limit
                    logger.info(f"Reached {file_size}MB limit for %s, stopping...", dataset_name)
                    break

        logger.info("Saved %s to %s", dataset_name, output_file)

    except Exception as e:
        logger.error("Error processing %s: %s", dataset_name, e)
        raise


def main():
    try:
        logger.info("🚀 Starting data import...")
        params = load_params(params_path=os.path.join(BASE_DIR, 'params.yaml'))
        file_size = params['data_import']['file_size']
        # file_size = 50

        process_dataset("ajibawa-2023/Children-Stories-Collection", file_size, L1_FILE, ["text"])
        process_dataset("bookcorpus", file_size, L2_FILE, ["text"])
        process_dataset("abisee/cnn_dailymail",file_size, L3_FILE, ["article", "highlights"], dataset_config="3.0.0")
        process_dataset("claran/modular-s2orc", file_size, L4_FILE, ["text"], dataset_config="ComputerScience,2019-2019")
        logger.info("✅ All datasets processed successfully!")
    except Exception as e:
        logger.error("Data import failed: %s", e)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
