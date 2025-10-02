import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import sys
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from typing import Tuple


# Use a small, fast model. "distilgpt2" is a good starter.
MODEL_NAME = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ----------------------------
# 1) Perplexity scorer (distilgpt2 for speed)
# ----------------------------
@torch.no_grad()
def perplexity(text: str, stride: int = 512, max_total_tokens: int = 10000) -> float:
    # 🛠️ Pre-truncate very long texts to avoid memory issues
    enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    
    max_len = getattr(model.config, "n_positions", 1024)
    n_tokens = input_ids.size(1)
    
    # If text is extremely long, truncate from the beginning
    if n_tokens > max_total_tokens:
        input_ids = input_ids[:, -max_total_tokens:]
        n_tokens = max_total_tokens
    
    if n_tokens == 0:
        return float("inf")

    lls = []
    total_tokens = 0

    for i in range(0, n_tokens, stride):
        begin_loc = max(i + stride - max_len, 0)
        end_loc = min(i + stride, n_tokens)

        input_chunk = input_ids[:, begin_loc:end_loc]
        
        # 🛠️ Double-check length constraint
        if input_chunk.size(1) > max_len:
            input_chunk = input_chunk[:, -max_len:]
        
        trg_len = input_chunk.size(1)
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_chunk, labels=target_ids)
        lls.append(outputs.loss * trg_len)
        total_tokens += trg_len

    ppl = torch.exp(torch.stack(lls).sum() / total_tokens)
    return ppl.item()


def build_dataset(n_imdb: int = 2000,
                  vocab=None,
                  k_words: int = 80,
                  n_random: int = 2000,
                  seed: int = 42) -> Tuple[list, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    if vocab is None:
        vocab = ["apple", "dog", "run", "blue", "pizza", "sky", "tree", "river", "phone",
                 "music", "street", "car", "book", "sound", "green", "water", "light", "chair"]

    # natural
    imdb = load_dataset("imdb", split=f"train[:{n_imdb}]")
    natural_texts = [x["text"] for x in imdb]
    y_natural = [0] * len(natural_texts)

    # random word salad
    random_texts = [" ".join(random.choices(vocab, k=k_words)) for _ in range(n_random)]
    y_random = [1] * len(random_texts)

    texts = natural_texts + random_texts
    labels = np.array(y_natural + y_random, dtype=int)

    # optional: shuffle to mix classes
    idx = np.arange(len(texts))
    np.random.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = labels[idx]
    return texts, labels

# ----------------------------
# 3) Threshold selection helpers
# ----------------------------
def best_threshold_by_f1(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # thresholds has length N-1; align with P/R
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    i = np.argmax(f1s)
    return thresholds[i], float(f1s[i]), float(precision[i]), float(recall[i])

def threshold_youden_j(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    i = np.argmax(j)
    return thr[i], float(tpr[i]), float(fpr[i]), float(roc_auc_score(y_true, scores))

def evaluate_at_threshold(y_true, scores, thr: float, title: str):
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = f1_score(y_true, y_pred)
    print(f"\n=== {title} ===")
    print(f"Threshold: {thr:.4f} | F1: {f1:.4f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(classification_report(y_true, y_pred, digits=3))



# texts, labels = data()
# sys.exit()
# scores = [perplexity(t) for t in texts]  # higher = more random-like
# print(scores)

# # Examples
# natural = "The pizza was great but the service was a bit slow; I'd still recommend it."
# random_soup = "Error detected: crash outage unresponsive system. Timeout overload corruption fault warning. Critical bug abnormal interference denied access. Failure unstable reboot misconfiguration. Deadlock insecure dependency anomaly compromised integrity broken recovery disrupted resilience."
# print("PPL natural   :", perplexity(natural))
# print("PPL random    :", perplexity(random_soup))


# ----------------------------
# 4) Main
# ----------------------------
if __name__ == "__main__":
    # Build data
    texts, labels = build_dataset(n_imdb=50, n_random=50)  # tweak sizes if slow

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.3, stratify=labels, random_state=123)

    # Score (perplexity is the score; higher => more random-like)
    print("Scoring validation set with perplexity...")
    val_scores = np.array([perplexity(t) for t in X_tr])


    # Pick thresholds
    thr_f1, best_f1, pre, rec = best_threshold_by_f1(y_tr, val_scores)
    thr_j, tpr_j, fpr_j, auc = threshold_youden_j(y_tr, val_scores)

    print("\n--- Thresholds from validation ---")
    print(f"F1-opt threshold: {thr_f1:.4f} | F1={best_f1:.4f} | Precision={pre:.4f} | Recall={rec:.4f}")
    print(f"Youden-J threshold: {thr_j:.4f} | AUC={auc:.4f} | TPR={tpr_j:.4f} | FPR={fpr_j:.4f}")

    # Evaluate both on the held-out test set
    print("\nScoring test set...")
    test_scores = np.array([perplexity(t) for t in X_te])

    evaluate_at_threshold(y_te, test_scores, thr_f1, "Test @ F1-opt threshold")
    evaluate_at_threshold(y_te, test_scores, thr_j,  "Test @ Youden-J threshold")

    # Save the chosen threshold (pick one policy)
    chosen_thr = float(thr_f1)  # or thr_j
    with open("random_detector_threshold.txt", "w") as f:
        f.write(str(chosen_thr))
    print(f"\nSaved threshold to random_detector_threshold.txt: {chosen_thr:.4f}")



