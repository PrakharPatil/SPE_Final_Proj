import yaml
import os
import torch
import joblib
import logging
from torch.optim import AdamW
from TransformerMS.Model.model_architecture import GPTLanguageModel
# from Model.model_architecture import GPTLanguageModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------------------- Setup Paths ----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------- Logging Setup ----------------------
log_dir = os.path.join(BASE_DIR, 'Logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("transformer_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "transformer_training.log"))
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

params = load_params(params_path=os.path.join(BASE_DIR, 'params.yaml'))
batch_size   = int(params["transformer_training"]["batch_size"])
block_size   = int(params["transformer_training"]["block_size"])
max_iters    = int(params["transformer_training"]["max_iters"])
eval_interval = int(params["transformer_training"]["eval_interval"])
learning_rate = float(params["transformer_training"]["learning_rate"])
eval_iters    = int(params["transformer_training"]["eval_iters"])
n_embd        = int(params["transformer_training"]["n_embd"])
n_head        = int(params["transformer_training"]["n_head"])
n_layer       = int(params["transformer_training"]["n_layer"])
dropout       = float(params["transformer_training"]["dropout"])
tokenizer = joblib.load(os.path.join(BASE_DIR, "Joblibs", "tokenizer.joblib"))

class Config:
    vocab_size = tokenizer.get_vocab_size()
    batch_size = batch_size
    block_size = block_size
    max_iters = max_iters
    eval_interval = eval_interval
    learning_rate = learning_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = eval_iters
    n_embd = n_embd
    n_head = n_head
    n_layer = n_layer
    dropout = dropout
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")


# ---------------------- Utilities ----------------------
def get_batch(data):
    ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i + Config.block_size] for i in ix]).to(Config.device)
    y = torch.stack([data[i + 1:i + Config.block_size + 1] for i in ix]).to(Config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {'train': [], 'val': []}
    for split, data in zip(['train', 'val'], [train_data, val_data]):
        for _ in range(Config.eval_iters):
            xb, yb = get_batch(data)
            _, loss = model(xb, yb)
            out[split].append(loss.item())
    model.train()
    return {k: sum(v) / len(v) for k, v in out.items()}

def compute_perplexity(loss):
    return torch.exp(torch.tensor(loss))


# ---------------------- Training Function ----------------------
def train_model(level_texts, tokenizer):
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    model = GPTLanguageModel(
        vocab_size=Config.vocab_size,
        block_size=Config.block_size,
        n_embd=Config.n_embd,
        n_head=Config.n_head,
        n_layer=Config.n_layer,
        dropout=Config.dropout
    ).to(Config.device)

    all_metrics = {
        "level": [], "iters": [],
        "train_loss": [], "val_loss": [],
        "train_ppl": [], "val_ppl": []
    }

    for level_i, text in enumerate(level_texts, start=1):
        logger.info(f"--- Training on Level {level_i} ---")
        encoded = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long).to(Config.device)
        split_idx = int(0.9 * len(encoded))
        train_data, val_data = encoded[:split_idx], encoded[split_idx:]

        optimizer = AdamW(model.parameters(), lr=Config.learning_rate)

        for it in range(Config.max_iters):
            if it % Config.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data)
                train_ppl = compute_perplexity(losses['train'])
                val_ppl = compute_perplexity(losses['val'])

                all_metrics["level"].append(level_i)
                all_metrics["iters"].append(it)
                all_metrics["train_loss"].append(losses["train"])
                all_metrics["val_loss"].append(losses["val"])
                all_metrics["train_ppl"].append(train_ppl.item())
                all_metrics["val_ppl"].append(val_ppl.item())

                logger.info(f"Iter {it:4d} | Train Loss: {losses['train']:.4f} "
                            f"(PPL: {train_ppl:.2f}) | Val Loss: {losses['val']:.4f} "
                            f"(PPL: {val_ppl:.2f})")

            xb, yb = get_batch(train_data)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint_path = os.path.join(Config.checkpoint_dir, f"model_level{level_i}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved model checkpoint at {checkpoint_path}")

        # Sample output
        context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
        sample = model.generate(context, max_new_tokens=300)[0].tolist()
        logger.info(f"Sample Output:\n{tokenizer.decode(sample)}")

    return model, all_metrics


# ---------------------- Main Entry ----------------------
if __name__ == "__main__":
    try:
        levels = []
        for i in range(1, 5):
            with open(os.path.join(BASE_DIR, "Data", "Clean", f"L{i}_cleaned.txt"), 'r', encoding='utf-8') as f:
                levels.append(f.read())

        model ,all_metrics = train_model(levels, tokenizer)
        # Save model
        joblib.dump(model, os.path.join(BASE_DIR, 'Joblibs', 'transformer.joblib'))
        # vocab_size = tokenizer.get_vocab_size(),
        # Save final model
        torch.save(dict(model_state_dict=model.state_dict(),
                        config=dict(vocab_size=Config.vocab_size, block_size=Config.block_size, n_embd=Config.n_embd,
                                    n_head=Config.n_head, n_layer=Config.n_layer, dropout=Config.dropout)), os.path.join(BASE_DIR, 'Joblibs', 'gpt_model.pth'))



    except Exception as e:
        logger.exception("Training failed due to an error.")
