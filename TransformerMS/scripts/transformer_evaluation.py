import json
import os
import sys
import time
import torch
import joblib
import logging
import sacrebleu
import yaml

from transformers import BartTokenizer, BartForConditionalGeneration

# ---------------------- Setup Paths ----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# ---------------------- Logging Setup ----------------------
log_dir = os.path.join(BASE_DIR, 'Logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("transformer_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "transformer_evaluation.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------- Configuration ----------------------
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load parameters: %s", e)
        raise

params = load_params(os.path.join(BASE_DIR, "params.yaml"))
tokenizer = joblib.load(os.path.join(BASE_DIR, "Joblibs", "tokenizer.joblib"))

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_size = int(params["transformer_evaluation"]["block_size"])
    eval_length = 50  # Number of tokens to compare

# ---------------------- Load BART for Scoring ----------------------
bart_model_name = 'facebook/bart-large-cnn'
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(Config.device)
bart_model.eval()

# ---------------------- Scoring Functions ----------------------
def compute_bleu(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def compute_bart_score(references, hypotheses):
    scores = []
    for ref, hyp in zip(references, hypotheses):
        input_ids = bart_tokenizer(hyp, return_tensors="pt").input_ids.to(Config.device)
        with bart_tokenizer.as_target_tokenizer():
            target_ids = bart_tokenizer(ref, return_tensors="pt").input_ids.to(Config.device)

        with torch.no_grad():
            output = bart_model(input_ids=input_ids, labels=target_ids)
            log_likelihood = -output.loss * target_ids.size(1)
            scores.append(log_likelihood.item())

    return sum(scores) / len(scores)

# ---------------------- Main Evaluation Function ----------------------
def evaluate_model(model, full_text):
    logger.info("Starting model evaluation...")

    try:
        # Step 1: Extract validation portion
        start = time.time()
        val_text = full_text[int(0.9 * len(full_text)):]
        logger.debug(f"Validation slicing took {time.time() - start:.2f} seconds")

        # Step 2: Tokenize
        start = time.time()
        val_ids = tokenizer.encode(val_text).ids
        logger.debug(f"Tokenizing took {time.time() - start:.2f} seconds")

        # Step 3: Tensor conversion
        start = time.time()
        val_data = torch.tensor(val_ids, dtype=torch.long).to(Config.device)
        logger.debug(f"Tensor creation took {time.time() - start:.2f} seconds")

        # Step 4: Prepare reference
        ref_ids = val_data[:Config.eval_length].tolist()
        reference = tokenizer.decode(ref_ids)

        # Step 5: Generate from model
        context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
        sample_ids = model.generate(context, max_new_tokens=Config.eval_length)[0].tolist()
        generated = tokenizer.decode(sample_ids)

        logger.info(f"\nReference:\n{reference}\n\nGenerated:\n{generated}")

        # Step 6: Compute Scores
        bleu = compute_bleu([reference], [generated])
        bart_score = compute_bart_score([reference], [generated])

        logger.info(f"\nBLEU Score:     {bleu:.2f}")
        logger.info(f"BARTScore (avg): {bart_score:.2f}")

        return {
            "bleu": bleu,
            "bart_score": bart_score,
            "reference": reference,
            "generated": generated
        }

    except Exception as e:
        logger.exception("Model evaluation failed.")
        raise

# ---------------------- Main Entry ----------------------
if __name__ == "__main__":
    try:
        logger.info("Loading model and tokenizer...")

        from TransformerMS.Model.model_architecture import GPTLanguageModel  # or correct path

        # Load model config and weights
        model_checkpoint = torch.load(os.path.join(BASE_DIR, 'Joblibs', 'gpt_model.pth'), map_location=Config.device)
        model_config = model_checkpoint['config']
        model = GPTLanguageModel(**model_config).to(Config.device)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        model.eval()
        logger.info("Model successfully loaded.")

        # Load full text
        with open(os.path.join(BASE_DIR, "Data", "Clean", "L4_cleaned.txt"), 'r', encoding='utf-8') as f:
            full_text = f.read()

        results = evaluate_model(model, full_text)

        # ---------------------- Save Evaluation Metrics ----------------------
        eval_dir = os.path.join(BASE_DIR, "Evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        metrics_path = os.path.join(eval_dir, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "BLEU": round(results["bleu"], 2),
                "BARTScore": round(results["bart_score"], 2)
            }, f)

        logger.info(f"Saved evaluation metrics to {metrics_path}")

    except Exception as e:
        logger.exception("Evaluation process failed.")
