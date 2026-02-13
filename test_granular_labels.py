"""
Test the continuous bias score and granular labeling system.
Samples 100 articles from the ABP dataset, runs them through the model,
and reports the distribution of the 5 granular labels.
"""

import json
import random
import re
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Load model
# ============================================================
print("=" * 60)
print("GRANULAR LABEL DISTRIBUTION TEST")
print("=" * 60)

model_path = "./bias-classifier-abp-final"
print(f"\nLoading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("  Model loaded.")

# ============================================================
# Source stripping (must match training)
# ============================================================
SOURCES_TO_STRIP = [
    'vox', 'vox.com', 'vox media',
    'vice', 'vice news', 'vice.com',
    'huffington post', 'huffpost', 'huff post', 'huffingtonpost',
    'buzzfeed', 'buzzfeed news', 'buzzfeednews',
    'guardian', 'the guardian', 'theguardian',
    'new york times', 'nyt', 'nytimes', 'the new york times',
    'reuters', 'associated press', 'ap news', 'ap',
    'bbc', 'bbc news',
    'business insider', 'businessinsider',
    'the hill', 'thehill',
    'npr', 'national public radio',
    'usa today', 'usatoday',
    'fox news', 'foxnews', 'fox',
    'new york post', 'ny post', 'nypost',
    'national review', 'nationalreview',
    'washington times', 'the washington times',
    'breitbart', 'breitbart news',
    'associated press contributed', 'reporting by', 'editing by',
    'compiled by', 'written by',
]

def strip_source_identifiers(text):
    cleaned = text
    for source in SOURCES_TO_STRIP:
        pattern = r'\b' + re.escape(source) + r'\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    return cleaned

# ============================================================
# Bias score computation (mirrors index.html logic)
# ============================================================
def compute_bias_score(left_prob, center_prob, right_prob):
    score = (-1 * left_prob) + (0 * center_prob) + (1 * right_prob)

    if score <= -0.6:
        label = "Left"
    elif score <= -0.2:
        label = "Left-Leaning"
    elif score <= 0.2:
        label = "Center"
    elif score <= 0.6:
        label = "Right-Leaning"
    else:
        label = "Right"

    return score, label

# ============================================================
# Load 100 random articles from ABP test set
# ============================================================
print("\nLoading ABP test set articles...")

splits_dir = Path("./abp-dataset/data/splits/random")
json_dir = Path("./abp-dataset/data/jsons")

test_ids = {}
with open(splits_dir / "test.tsv", 'r') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            test_ids[parts[0]] = int(parts[1])

id2label_true = {0: 'left', 1: 'center', 2: 'right'}

# Sample 100 random articles
random.seed(42)
sampled_ids = random.sample(list(test_ids.keys()), min(100, len(test_ids)))

articles = []
for article_id in sampled_ids:
    json_path = json_dir / f"{article_id}.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = data.get('content_original', data.get('content', ''))
        if len(text) >= 100:
            articles.append({
                'id': article_id,
                'text': text,
                'true_label': id2label_true[test_ids[article_id]],
                'source': data.get('source', ''),
                'title': data.get('title', '')[:80],
            })

print(f"  Loaded {len(articles)} articles from test set")

# True label distribution in sample
true_dist = {}
for a in articles:
    true_dist[a['true_label']] = true_dist.get(a['true_label'], 0) + 1
print(f"\n  True label distribution in sample:")
for lbl in ['left', 'center', 'right']:
    print(f"    {lbl}: {true_dist.get(lbl, 0)}")

# ============================================================
# Run predictions
# ============================================================
print(f"\n{'=' * 60}")
print("RUNNING PREDICTIONS ON 100 ARTICLES")
print(f"{'=' * 60}")

granular_counts = {"Left": 0, "Left-Leaning": 0, "Center": 0, "Right-Leaning": 0, "Right": 0}
scores_all = []
results_detail = []

with torch.no_grad():
    for i, article in enumerate(articles):
        text = strip_source_identifiers(article['text'])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        left_prob = float(probs[0][0])
        center_prob = float(probs[0][1])
        right_prob = float(probs[0][2])

        score, granular_label = compute_bias_score(left_prob, center_prob, right_prob)

        granular_counts[granular_label] += 1
        scores_all.append(score)
        results_detail.append({
            'true': article['true_label'],
            'granular': granular_label,
            'score': score,
            'left_p': left_prob,
            'center_p': center_prob,
            'right_p': right_prob,
            'title': article['title'],
        })

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(articles)} articles...")

# ============================================================
# Results
# ============================================================
print(f"\n{'=' * 60}")
print("GRANULAR LABEL DISTRIBUTION (100 articles)")
print(f"{'=' * 60}")

total = len(articles)
for label in ["Left", "Left-Leaning", "Center", "Right-Leaning", "Right"]:
    count = granular_counts[label]
    bar = "#" * int(count / total * 50)
    print(f"  {label:>15}: {count:3d} ({count/total*100:5.1f}%)  {bar}")

print(f"\n  Total: {total}")

# Score statistics
scores_arr = np.array(scores_all)
print(f"\n{'=' * 60}")
print("CONTINUOUS SCORE STATISTICS")
print(f"{'=' * 60}")
print(f"  Mean:   {scores_arr.mean():+.3f}")
print(f"  Median: {np.median(scores_arr):+.3f}")
print(f"  Std:    {scores_arr.std():.3f}")
print(f"  Min:    {scores_arr.min():+.3f}")
print(f"  Max:    {scores_arr.max():+.3f}")

# Score distribution by true label
print(f"\n{'=' * 60}")
print("MEAN SCORE BY TRUE LABEL")
print(f"{'=' * 60}")
for true_label in ['left', 'center', 'right']:
    label_scores = [r['score'] for r in results_detail if r['true'] == true_label]
    if label_scores:
        mean_s = np.mean(label_scores)
        print(f"  True {true_label:>6}: mean score = {mean_s:+.3f}  (n={len(label_scores)})")

# Cross-tabulation: true label vs granular label
print(f"\n{'=' * 60}")
print("TRUE LABEL vs GRANULAR LABEL CROSS-TAB")
print(f"{'=' * 60}")

granular_labels_order = ["Left", "Left-Leaning", "Center", "Right-Leaning", "Right"]
print(f"{'':>10}", end="")
for gl in granular_labels_order:
    print(f"  {gl:>13}", end="")
print()

for true_label in ['left', 'center', 'right']:
    print(f"  {true_label:>6}  ", end="")
    for gl in granular_labels_order:
        count = sum(1 for r in results_detail if r['true'] == true_label and r['granular'] == gl)
        print(f"  {count:>13}", end="")
    print()

# Sample articles from each granular category
print(f"\n{'=' * 60}")
print("SAMPLE ARTICLES PER GRANULAR LABEL")
print(f"{'=' * 60}")

for gl in granular_labels_order:
    matches = [r for r in results_detail if r['granular'] == gl]
    if matches:
        sample = matches[0]
        print(f"\n  [{gl}] score={sample['score']:+.3f} (true: {sample['true']})")
        print(f"    L={sample['left_p']:.2f} C={sample['center_p']:.2f} R={sample['right_p']:.2f}")
        print(f"    Title: {sample['title']}")

print(f"\n{'=' * 60}")
