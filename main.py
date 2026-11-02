#!/usr/bin/env python3
"""
Project 49: Fine-Tune BERT for Sentiment Analysis
==================================================
Fine-tunes a pretrained BERT (Bidirectional Encoder Representations from
Transformers) model on IMDb movie reviews for binary sentiment classification.

Key Highlights:
  - Hugging Face `bert-base-uncased` (110M parameters, 12 layers, 12 heads, 768 hidden).
  - Fine-tuning with AdamW optimizer, linear warmup, and weight decay.
  - Multi-head self-attention interpretability: extracts [CLS] token attention weights
    across heads in the final encoder layer to visualize token saliency.
  - Comprehensive metric evaluation: Accuracy, Precision, Recall, F1, ROC-AUC,
    and 2x2 Confusion Matrix.
  - 4-Panel diagnostic visual artifact.
  - Interactive CLI sentiment analysis tool.
"""

import os
import sys
import json
import time
import math
import argparse
from typing import List, Dict, Tuple, Any

# Suppress Hugging Face symlink warnings on Windows & enforce offline mode
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "data", "imdb_sentiment.json")
DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "bert_sentiment_best.pt")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "outputs", "bert_sentiment_results.png")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")


# ==============================================================================
# 1. Dataset & Data Loading
# ==============================================================================

class IMDbDataset(Dataset):
    """PyTorch Dataset for tokenized IMDb reviews."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def load_imdb_splits(json_path: str, tokenizer, max_length: int = 128):
    """Load train, val, test splits from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = {}
    for split_name in ["train", "validation", "test"]:
        items = data[split_name]
        texts = [x["text"] for x in items]
        labels = [x["label"] for x in items]
        splits[split_name] = IMDbDataset(texts, labels, tokenizer, max_length=max_length)
        splits[f"{split_name}_raw"] = items

    return splits


# ==============================================================================
# 2. Evaluation Metrics Suite
# ==============================================================================

def compute_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
    """Compute Area Under ROC Curve via trapezoidal integration."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    num_pos = sum(y_true)
    num_neg = len(y_true) - num_pos
    if num_pos == 0 or num_neg == 0:
        return 0.5

    tp, fp = 0, 0
    tpr_list, fpr_list = [0.0], [0.0]

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / num_pos)
        fpr_list.append(fp / num_neg)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        dx = fpr_list[i] - fpr_list[i - 1]
        y_avg = (tpr_list[i] + tpr_list[i - 1]) / 2.0
        auc += dx * y_avg
    return auc


def evaluate_classifier(model, dataloader, device) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, f1, roc-auc, and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n if n > 0 else 0.0

    tp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 1)

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = compute_roc_auc(all_labels, all_probs)

    conf_matrix = [[tn, fp], [fn, tp]]

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": conf_matrix,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


# ==============================================================================
# 3. Fine-Tuning Pipeline
# ==============================================================================

class BERTSentimentClassifier:
    """Manages BERT loading, fine-tuning, interpretability, and inference."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[*] Initializing BERT Sentiment Classifier on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            attn_implementation="eager",
            local_files_only=True
        )
        self.model.to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[+] Loaded '{model_name}' ({param_count:,} params, {trainable:,} trainable).")

    def fine_tune(
        self,
        data_splits,
        epochs: int = 4,
        batch_size: int = 4,
        lr: float = 2e-5,
        checkpoint_path: str = DEFAULT_CHECKPOINT
    ) -> Dict[str, List[float]]:
        """Fine-tune BERT with AdamW and track training dynamics."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        train_loader = DataLoader(data_splits["train"], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(data_splits["validation"], batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        print("\n" + "=" * 80)
        print(f"FINE-TUNING BERT FOR SENTIMENT ANALYSIS (Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr})")
        print("=" * 80)

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            self.model.train()

            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size_cur = labels.size(0)
                epoch_loss += loss.item() * batch_size_cur
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += batch_size_cur

            train_loss = epoch_loss / total
            train_acc = correct / total

            # Validation step
            val_eval = evaluate_classifier(self.model, val_loader, self.device)
            val_loss = val_eval["loss"]
            val_acc = val_eval["accuracy"]
            dt = time.perf_counter() - t0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.2f}s) | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  [+] Saved new optimal checkpoint to: {checkpoint_path}")

        # Load best weights
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"\n[+] Restored best validation model checkpoint.")

        return history

    def predict(self, text: str, extract_attention: bool = False) -> Dict[str, Any]:
        """Perform sentiment inference on a single text string."""
        self.model.eval()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=extract_attention)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pos_prob = probs[1].item()
        neg_prob = probs[0].item()
        pred_label = int(torch.argmax(logits, dim=-1).item())
        sentiment = "POSITIVE" if pred_label == 1 else "NEGATIVE"
        confidence = max(pos_prob, neg_prob) * 100.0

        res = {
            "text": text,
            "sentiment": sentiment,
            "label": pred_label,
            "confidence": confidence,
            "pos_prob": pos_prob,
            "neg_prob": neg_prob,
        }

        if extract_attention and outputs.attentions is not None:
            # Last layer attention: (batch=1, num_heads=12, seq_len, seq_len)
            # Head-averaged attention from [CLS] (token index 0) to all tokens
            last_layer_attn = outputs.attentions[-1][0]  # (12, seq_len, seq_len)
            cls_attn = last_layer_attn.mean(dim=0)[0].cpu().numpy()  # (seq_len,)

            token_ids = inputs.input_ids[0].cpu().tolist()
            tokens = [self.tokenizer.decode([tid]).strip() for tid in token_ids]

            res["tokens"] = tokens
            res["cls_attention"] = cls_attn

        return res


# ==============================================================================
# 4. Diagnostic Visualization Suite
# ==============================================================================

def generate_visualization(
    history: Dict[str, List[float]],
    test_eval: Dict[str, Any],
    sample_attn_res: Dict[str, Any],
    output_path: str = DEFAULT_OUTPUT
):
    """
    Generate professional 4-panel diagnostic figure using pure Matplotlib:
      (a) Training & Validation Loss & Accuracy Curves
      (b) Annotated 2x2 Confusion Matrix & Test Metrics Box
      (c) Prediction Probability Distribution (Positive vs Negative Reviews)
      (d) Token-Level Saliency Attention Map from [CLS] Token
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # -------------------------------------------------------------------------
    # Panel 1: Learning Trajectories (Loss & Accuracy)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    color_loss = "#d62728"
    ax1.plot(epochs, history["train_loss"], color=color_loss, linestyle="--", marker="o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], color=color_loss, linewidth=2.5, marker="s", label="Val Loss")
    ax1.set_xlabel("Fine-Tuning Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", color=color_loss, fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    color_acc = "#1f77b4"
    ax1_twin.plot(epochs, [a * 100 for a in history["train_acc"]], color=color_acc, linestyle="--", marker="^", label="Train Acc")
    ax1_twin.plot(epochs, [a * 100 for a in history["val_acc"]], color=color_acc, linewidth=2.5, marker="D", label="Val Acc")
    ax1_twin.set_ylabel("Accuracy (%)", color=color_acc, fontsize=10, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_acc)
    ax1_twin.set_ylim(40, 105)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.9)
    ax1.set_title("BERT Fine-Tuning Convergence Trajectories", fontsize=11, fontweight="bold")
    ax1.set_xticks(epochs)

    # -------------------------------------------------------------------------
    # Panel 2: Confusion Matrix & Metrics Summary
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    cm = np.array(test_eval["confusion_matrix"])
    im2 = ax2.imshow(cm, cmap="Blues", interpolation="nearest")

    classes = ["Negative (0)", "Positive (1)"]
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax2.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax2.set_ylabel("True Class", fontsize=10, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax2.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=16, fontweight="bold")

    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    metrics_text = (
        f"Test Accuracy : {test_eval['accuracy']*100:.1f}%\n"
        f"Precision     : {test_eval['precision']*100:.1f}%\n"
        f"Recall        : {test_eval['recall']*100:.1f}%\n"
        f"F1-Score      : {test_eval['f1']*100:.1f}%\n"
        f"ROC-AUC       : {test_eval['roc_auc']:.4f}"
    )
    ax2.text(1.25, 0.5, metrics_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="center", bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#6c757d"))
    ax2.set_title("Test Set Confusion Matrix & Performance Metrics", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Prediction Probability Distribution
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    probs = test_eval["probabilities"]
    labels = test_eval["labels"]

    pos_probs = [p for p, y in zip(probs, labels) if y == 1]
    neg_probs = [p for p, y in zip(probs, labels) if y == 0]

    indices_pos = np.arange(len(pos_probs))
    indices_neg = np.arange(len(neg_probs))

    ax3.bar(indices_pos - 0.2, pos_probs, 0.4, label="True Positive Reviews", color="#2ca02c", alpha=0.85, edgecolor="black")
    ax3.bar(indices_neg + 0.2, neg_probs, 0.4, label="True Negative Reviews", color="#d62728", alpha=0.85, edgecolor="black")
    ax3.axhline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Boundary (0.5)")

    ax3.set_xlabel("Review Sample Index", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Predicted Positive Probability $P(Y=1|X)$", fontsize=10, fontweight="bold")
    ax3.set_title("Confidence Calibration: Softmax Class Probabilities", fontsize=11, fontweight="bold")
    ax3.set_ylim(-0.05, 1.1)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax3.legend(loc="upper right", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Token-Level Attention Saliency from [CLS]
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    if sample_attn_res and "cls_attention" in sample_attn_res:
        tokens = sample_attn_res["tokens"]
        attn = sample_attn_res["cls_attention"]

        # Truncate to first 18 tokens for readability
        max_tok = min(18, len(tokens))
        sub_tokens = [t if t else "<sp>" for t in tokens[:max_tok]]
        sub_attn = attn[:max_tok]

        colors = plt.cm.plasma(sub_attn / max(sub_attn.max(), 1e-6))
        ax4.bar(range(max_tok), sub_attn, color=colors, edgecolor="black", alpha=0.9)
        ax4.set_xticks(range(max_tok))
        ax4.set_xticklabels(sub_tokens, rotation=55, ha="right", fontsize=9, fontweight="bold")
        ax4.set_ylabel("Attention Weight from [CLS]", fontsize=10, fontweight="bold")
        ax4.set_title(f"Interpretability: [CLS] Self-Attention ({sample_attn_res['sentiment']})", fontsize=11, fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.4, axis="y")
    else:
        ax4.text(0.5, 0.5, "Attention Map\nUnavailable", ha="center", va="center", fontsize=12)
        ax4.set_title("Interpretability: [CLS] Self-Attention", fontsize=11, fontweight="bold")

    plt.suptitle("Fine-Tuned BERT: IMDb Sentiment Classification & Saliency Analysis", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved diagnostic visualization figure: {output_path}")


# ==============================================================================
# 5. Main Benchmark Pipeline
# ==============================================================================

def run_benchmark(data_path: str = DEFAULT_DATA, epochs: int = 4, batch_size: int = 4, lr: float = 2e-5) -> None:
    """Execute complete BERT fine-tuning, evaluation, and interpretability benchmark."""
    classifier = BERTSentimentClassifier()
    splits = load_imdb_splits(data_path, classifier.tokenizer)

    # Train / Fine-tune
    history = classifier.fine_tune(splits, epochs=epochs, batch_size=batch_size, lr=lr)

    # Evaluate on held-out test split
    print("\n" + "=" * 80)
    print("EVALUATING FINE-TUNED BERT ON HELD-OUT TEST SPLIT (N=8)")
    print("=" * 80)

    test_loader = DataLoader(splits["test"], batch_size=batch_size, shuffle=False)
    test_eval = evaluate_classifier(classifier.model, test_loader, classifier.device)

    print(f"  - Test Loss     : {test_eval['loss']:.4f}")
    print(f"  - Test Accuracy : {test_eval['accuracy']*100:.2f}%")
    print(f"  - Precision     : {test_eval['precision']*100:.2f}%")
    print(f"  - Recall        : {test_eval['recall']*100:.2f}%")
    print(f"  - F1-Score      : {test_eval['f1']*100:.2f}%")
    print(f"  - ROC-AUC       : {test_eval['roc_auc']:.4f}")

    # Interpretability sample
    sample_text = "An extraordinary achievement in modern cinema, delivering stunning visuals, poignant themes, and brilliant acting."
    sample_res = classifier.predict(sample_text, extract_attention=True)

    # Generate Visualization
    generate_visualization(history, test_eval, sample_res)

    print("\n" + "=" * 80)
    print("SAMPLE INFERENCE & INTERPRETABILITY:")
    print(f"  Review    : \"{sample_text}\"")
    print(f"  Prediction: {sample_res['sentiment']} ({sample_res['confidence']:.2f}% confidence)")
    print("=" * 80 + "\n")


def interactive_analyzer():
    """Interactive console prompt for user movie review sentiment classification."""
    classifier = BERTSentimentClassifier()
    if os.path.exists(DEFAULT_CHECKPOINT):
        classifier.model.load_state_dict(torch.load(DEFAULT_CHECKPOINT, map_location=classifier.device))
        print("[+] Loaded fine-tuned checkpoint.")

    print("\n" + "=" * 80)
    print("Interactive BERT Sentiment Analyzer (Enter 'q' to quit)")
    print("=" * 80)

    while True:
        try:
            review = input("\nEnter movie review: ").strip()
            if review.lower() == "q":
                break
            if not review:
                continue

            res = classifier.predict(review)
            color = "\033[92m" if res["sentiment"] == "POSITIVE" else "\033[91m"
            print(f"Sentiment : {color}{res['sentiment']}\033[0m")
            print(f"Confidence: {res['confidence']:.2f}% (P(Pos)={res['pos_prob']:.3f}, P(Neg)={res['neg_prob']:.3f})\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.")
            break


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune BERT for IMDb Movie Review Sentiment Analysis")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to IMDb JSON dataset")
    parser.add_argument("--epochs", type=int, default=4, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--review", type=str, default="", help="Single review text to classify")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive sentiment console")
    args = parser.parse_args()

    if args.interactive:
        interactive_analyzer()
    elif args.review:
        classifier = BERTSentimentClassifier()
        if os.path.exists(DEFAULT_CHECKPOINT):
            classifier.model.load_state_dict(torch.load(DEFAULT_CHECKPOINT, map_location=classifier.device))
        res = classifier.predict(args.review)
        print("\n" + "=" * 80)
        print("REVIEW TEXT:")
        print(args.review)
        print("\nCLASSIFICATION RESULT:")
        print(f"Sentiment : {res['sentiment']}")
        print(f"Confidence: {res['confidence']:.2f}% (P(Pos): {res['pos_prob']:.4f}, P(Neg): {res['neg_prob']:.4f})")
        print("=" * 80 + "\n")
    else:
        run_benchmark(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()

