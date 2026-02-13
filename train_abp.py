"""
Train DistilBERT on the Article Bias Prediction dataset (37,554 articles).

This dataset has per-article left/center/right labels assigned by AllSides editors,
unlike the original HF dataset which used per-outlet labels.

Source identifiers are stripped from article text to prevent the model from
learning outlet-specific shortcuts instead of genuine bias signals.
"""

import json
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report, confusion_matrix
import torch

print("=" * 60)
print("TRAINING ON ARTICLE BIAS PREDICTION DATASET")
print("=" * 60)
print("37,554 articles with per-article left/center/right labels")
print("from AllSides editors (not per-outlet labeling)")

# GPU check
print("\nGPU Check:")
if torch.cuda.is_available():
    print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("  [!] No GPU detected - training will be slow but will proceed")

# ============================================================
# Load ABP dataset from JSON files
# ============================================================
print("\nLoading Article Bias Prediction dataset...")

json_dir = Path("./abp-dataset/data/jsons")
splits_dir = Path("./abp-dataset/data/splits/random")

# Load split IDs
def load_split(split_file):
    ids = {}
    with open(split_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                ids[parts[0]] = int(parts[1])
    return ids

train_ids = load_split(splits_dir / "train.tsv")
valid_ids = load_split(splits_dir / "valid.tsv")
test_ids = load_split(splits_dir / "test.tsv")

print(f"  Split sizes: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}")

# Load article content from JSON files
def load_articles(id_dict, json_dir):
    records = []
    missing = 0
    for article_id, bias_label in id_dict.items():
        json_path = json_dir / f"{article_id}.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            records.append({
                'text': data.get('content_original', data.get('content', '')),
                'labels': bias_label,
                'source': data.get('source', ''),
                'topic': data.get('topic', ''),
                'title': data.get('title', ''),
                'date': data.get('date', ''),
            })
        else:
            missing += 1
    if missing > 0:
        print(f"  Warning: {missing} JSON files not found")
    return pd.DataFrame(records)

train_df = load_articles(train_ids, json_dir)
valid_df = load_articles(valid_ids, json_dir)
test_df = load_articles(test_ids, json_dir)

print(f"\n  Loaded: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

# ============================================================
# Dataset statistics
# ============================================================
id2label = {0: 'left', 1: 'center', 2: 'right'}
label2id = {'left': 0, 'center': 1, 'right': 2}

print(f"\n{'=' * 60}")
print("DATASET STATISTICS")
print(f"{'=' * 60}")

for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
    label_counts = df['labels'].value_counts().sort_index()
    label_names = {id2label[k]: v for k, v in label_counts.items()}
    print(f"\n  {name}: {len(df)} articles")
    for lbl, cnt in label_names.items():
        print(f"    {lbl}: {cnt} ({cnt/len(df)*100:.1f}%)")

# Source diversity
print(f"\n  Unique sources in train: {train_df['source'].nunique()}")
print(f"  Top 10 sources:")
for src, cnt in train_df['source'].value_counts().head(10).items():
    print(f"    {src}: {cnt}")

# ============================================================
# Source identifier stripping
# ============================================================
print(f"\n{'=' * 60}")
print("STRIPPING SOURCE IDENTIFIERS")
print(f"{'=' * 60}")

# Build comprehensive list from dataset sources + known outlets
dataset_sources = set()
for df in [train_df, valid_df, test_df]:
    dataset_sources.update(df['source'].dropna().unique())

SOURCES_TO_STRIP = [
    # From original training pipeline
    'vox', 'vox.com', 'vox media',
    'vice', 'vice news', 'vice.com',
    'huffington post', 'huffpost', 'huff post', 'huffingtonpost',
    'buzzfeed', 'buzzfeed news', 'buzzfeednews',
    'guardian', 'the guardian', 'theguardian',
    'new york times', 'nyt', 'nytimes', 'the new york times',
    'reuters', 'associated press', 'ap news',
    'bbc', 'bbc news',
    'business insider', 'businessinsider',
    'the hill', 'thehill',
    'npr', 'national public radio',
    'usa today', 'usatoday',
    'fox news', 'foxnews',
    'new york post', 'ny post', 'nypost',
    'national review', 'nationalreview',
    'washington times', 'the washington times',
    'breitbart', 'breitbart news',
    # Additional outlets found in ABP dataset
    'washington post', 'the washington post',
    'cnn', 'msnbc',
    'daily caller', 'daily wire', 'daily beast',
    'the federalist', 'federalist',
    'slate', 'salon',
    'the atlantic', 'atlantic',
    'politico',
    'the daily signal',
    'the blaze', 'theblaze',
    'townhall',
    'reason', 'reason.com',
    'the intercept',
    'mother jones',
    'think progress', 'thinkprogress',
    'the american conservative',
    'washington examiner',
    'the epoch times',
    'newsweek',
    'time magazine',
    'chicago tribune',
    'los angeles times', 'la times',
    'wall street journal', 'wsj',
    'christian science monitor',
    'allsides',
    # Common boilerplate
    'associated press contributed', 'reporting by', 'editing by',
    'compiled by', 'written by',
]

# Also add the actual source names from the dataset
for src in dataset_sources:
    src_lower = src.lower().strip()
    if src_lower and src_lower not in [s.lower() for s in SOURCES_TO_STRIP]:
        SOURCES_TO_STRIP.append(src_lower)

def strip_source_identifiers(text):
    """Remove publication names and outlet-specific boilerplate from article text."""
    if not isinstance(text, str):
        return ''
    cleaned = text
    for source in SOURCES_TO_STRIP:
        pattern = r'\b' + re.escape(source) + r'\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    return cleaned

print("Stripping source identifiers from all splits...")
for df in [train_df, valid_df, test_df]:
    df['text'] = df['text'].apply(strip_source_identifiers)

# Filter out empty/very short articles
min_length = 100
for name, df_ref in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    before = len(df_ref)
    mask = df_ref['text'].str.len() >= min_length
    if name == "train":
        train_df = df_ref[mask].copy()
    elif name == "valid":
        valid_df = df_ref[mask].copy()
    else:
        test_df = df_ref[mask].copy()
    after = len(mask[mask])
    if before != after:
        print(f"  {name}: removed {before - after} articles shorter than {min_length} chars")

print(f"  Done. Final sizes: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

# ============================================================
# Tokenization
# ============================================================
print(f"\n{'=' * 60}")
print("TOKENIZING")
print(f"{'=' * 60}")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = Dataset.from_pandas(train_df[['text', 'labels']].reset_index(drop=True))
valid_dataset = Dataset.from_pandas(valid_df[['text', 'labels']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[['text', 'labels']].reset_index(drop=True))

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=512
    )

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
valid_dataset = valid_dataset.map(tokenize_function, batched=True, batch_size=1000)
test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=1000)

# ============================================================
# Model setup
# ============================================================
print(f"\n{'=' * 60}")
print("MODEL SETUP")
print(f"{'=' * 60}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# Training
# ============================================================
training_args = TrainingArguments(
    output_dir="./bias-classifier-abp",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    logging_dir='./logs-abp',
    logging_steps=100,
    save_total_limit=2,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = (predictions == labels).mean()

    per_class_acc = {}
    for label_id, label_name in id2label.items():
        mask = labels == label_id
        if mask.sum() > 0:
            per_class_acc[f'acc_{label_name}'] = float((predictions[mask] == labels[mask]).mean())

    return {
        'accuracy': float(accuracy),
        **per_class_acc
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Baseline
print(f"\n{'=' * 60}")
print("BASELINE (before training)")
print(f"{'=' * 60}")
baseline = trainer.evaluate()
for metric, value in baseline.items():
    if metric.startswith('eval_'):
        print(f"  {metric}: {value:.4f}")

# Train
print(f"\n{'=' * 60}")
print("TRAINING")
print(f"{'=' * 60}")
print("This will take 1-3 hours depending on GPU...\n")

trainer.train()

# ============================================================
# Evaluation on validation set
# ============================================================
print(f"\n{'=' * 60}")
print("VALIDATION SET EVALUATION")
print(f"{'=' * 60}")

eval_results = trainer.evaluate()
for metric, value in eval_results.items():
    if metric.startswith('eval_'):
        print(f"  {metric}: {value:.4f}")

# ============================================================
# Evaluation on test set
# ============================================================
print(f"\n{'=' * 60}")
print("TEST SET EVALUATION")
print(f"{'=' * 60}")

test_results = trainer.evaluate(test_dataset)
for metric, value in test_results.items():
    if metric.startswith('eval_'):
        print(f"  {metric}: {value:.4f}")

# Detailed test predictions
test_preds = trainer.predict(test_dataset)
y_pred = np.argmax(test_preds.predictions, axis=1)
y_true = test_preds.label_ids

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=['left', 'center', 'right']
))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("Confusion Matrix:")
print("Predicted ->     left  center  right")
for i, label in enumerate(['left', 'center', 'right']):
    print(f"  {label:>6}     {cm[i][0]:>5}  {cm[i][1]:>6}  {cm[i][2]:>5}")

# Prediction distribution check
pred_dist = pd.Series(y_pred).value_counts().sort_index()
print(f"\nPrediction distribution on test set:")
for label_id in [0, 1, 2]:
    count = pred_dist.get(label_id, 0)
    print(f"  {id2label[label_id]}: {count} ({count/len(y_pred)*100:.1f}%)")

# ============================================================
# Save model
# ============================================================
print(f"\n{'=' * 60}")
print("SAVING MODEL")
print(f"{'=' * 60}")

model_path = "./bias-classifier-abp-final"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"  Model saved to: {model_path}")

# Quick smoke test
print("\nSmoke test predictions:")
from transformers import pipeline
classifier = pipeline("text-classification", model=model_path, top_k=None)

test_texts = {
    "right": "The radical left socialist agenda is destroying our great nation and threatening our constitutional freedoms.",
    "center": "The Federal Reserve announced it will maintain interest rates at current levels following its two-day policy meeting.",
    "left": "Income inequality continues to widen as corporate profits reach record highs while workers struggle to make ends meet.",
}

for expected, text in test_texts.items():
    result = classifier(text)[0]
    predicted = max(result, key=lambda x: x['score'])
    predicted_label = predicted['label'].lower()
    match = "[OK]" if predicted_label == expected else "[X]"
    print(f"  {match} Expected {expected}, got {predicted_label} ({predicted['score']:.2f})")

print(f"\n{'=' * 60}")
print("TRAINING COMPLETE")
print(f"{'=' * 60}")
accuracy = eval_results.get('eval_accuracy', 0)
print(f"\n  Validation accuracy: {accuracy*100:.1f}%")
print(f"  Model saved to: {model_path}")
print(f"\n  To upload: update upload_to_huggingface.py to use '{model_path}'")
print(f"{'=' * 60}")
