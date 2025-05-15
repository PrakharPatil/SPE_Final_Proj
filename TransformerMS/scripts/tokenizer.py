import os
import logging
import joblib
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import yaml
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# -------------- Logger Setup --------------
log_dir = os.path.join(BASE_DIR, 'Logs')
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('tokenizer_stage')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'tokenizer_stage.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------- Paths --------------
# BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))

CLEAN_DIR = os.path.join(BASE_DIR, 'Data', 'Clean')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'Joblibs')
os.makedirs(TOKENIZER_PATH, exist_ok=True)

FILES = {
    'L1': os.path.join(CLEAN_DIR, 'L1_Cleaned.txt'),
    'L2': os.path.join(CLEAN_DIR, 'L2_Cleaned.txt'),
    'L3': os.path.join(CLEAN_DIR, 'L3_Cleaned.txt'),
    'L4': os.path.join(CLEAN_DIR, 'L4_Cleaned.txt')
}

# -------------- Helper Functions --------------
def load_cleaned_text(filepath: str) -> str:
    """Load cleaned text from file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        raise
    except Exception as e:
        logger.error("Failed to load text from %s: %s", filepath, e)
        raise

def get_stats(byte_data: bytes) -> dict:
    """Return frequency statistics for adjacent byte pairs."""
    try:
        counts = {}
        for pair in zip(byte_data, byte_data[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    except Exception as e:
        logger.error("Error in calculating byte pair stats: %s", e)
        raise

def chunked_docs(docs: list[str]):
    """Yield one line at a time from all cleaned documents."""
    for doc in docs:
        for line in doc.split('\n'):
            line = line.strip()
            if line:
                yield line

def train_tokenizer(docs: list[str], vocab_size=2000) -> Tokenizer:
    """Train and return a BPE tokenizer."""
    try:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
        tokenizer.train_from_iterator(chunked_docs(docs), trainer=trainer)
        logger.debug("Tokenizer trained with vocab size: %d", tokenizer.get_vocab_size())
        return tokenizer
    except Exception as e:
        logger.error("Failed to train tokenizer: %s", e)
        raise

def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """Serialize and save the tokenizer object using joblib."""
    try:
        joblib.dump(tokenizer, path)
        logger.debug("Tokenizer saved to %s", path)
    except Exception as e:
        logger.error("Failed to save tokenizer: %s", e)
        raise

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load parameters: %s", e)
        raise

# -------------- Main Pipeline --------------
def main():
    try:
        logger.info("Starting tokenizer stage...")

        # Load cleaned text files
        L1 = load_cleaned_text(FILES['L1'])
        L2 = load_cleaned_text(FILES['L2'])
        L3 = load_cleaned_text(FILES['L3'])
        L4 = load_cleaned_text(FILES['L4'])

        full_text = L1 + L2 + L3 + L4
        D = len(full_text)
        logger.debug("Total cleaned text length: %d characters", D)

        token = full_text.encode("utf-8")
        stats = get_stats(token)
        chars = sorted(list(set(full_text)))
        print("All Unique Cleaned Characters : ")
        print(''.join(chars))
        top_pair = max(stats, key=stats.get)
        logger.debug("Most frequent byte pair: %s", top_pair)

        # Train tokenizer
        params = load_params(params_path=os.path.join(BASE_DIR, 'params.yaml'))
        vocab_size = params['tokenizer']['vocab_size']
        tokenizer = train_tokenizer([L1, L2, L3, L4],vocab_size)

        # Save tokenizer
        save_tokenizer(tokenizer, os.path.join(TOKENIZER_PATH, "tokenizer.joblib"))

        # Test encode/decode
        sample = "math is beautiful ✨"
        ids = tokenizer.encode(sample).ids
        logger.debug("Sample encode → %s", ids)
        logger.debug("Sample decode ← %s", tokenizer.decode(ids))

        logger.info("Tokenizer stage completed successfully.")

    except Exception as e:
        logger.error("Tokenizer stage failed: %s", e)
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
