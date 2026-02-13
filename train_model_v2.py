"""
Fine-tune DistilBERT on 30k articles for production deployment

This model will be uploaded to Hugging Face Hub and called from GitHub Pages
"""

import pandas as pd
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
os.environ['HF_HUB_OFFLINE'] = '1'  # Use cached models only

print("="*60)
print("TRAINING PRODUCTION BIAS CLASSIFIER")
print("="*60)

# Verify GPU
print("\nGPU Check:")
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  ✗ WARNING: No GPU detected - training will be VERY slow")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit()

# ============================================================
# Source identifier stripping
# ============================================================
# Publication names, bylines, and outlet-specific boilerplate that
# let the model shortcut by recognizing the source instead of
# learning genuine content-level bias signals.

SOURCES_TO_STRIP = [
    # Left-labeled outlets
    'vox', 'vox.com', 'vox media',
    'vice', 'vice news', 'vice.com',
    'huffington post', 'huffpost', 'huff post', 'huffingtonpost',
    'buzzfeed', 'buzzfeed news', 'buzzfeednews',
    'guardian', 'the guardian', 'theguardian',
    'new york times', 'nyt', 'nytimes', 'the new york times',
    # Center-labeled outlets
    'reuters', 'associated press', 'ap news', 'ap',
    'bbc', 'bbc news',
    'business insider', 'businessinsider',
    'the hill', 'thehill',
    'npr', 'national public radio',
    'usa today', 'usatoday',
    # Right-labeled outlets
    'fox news', 'foxnews', 'fox',
    'new york post', 'ny post', 'nypost',
    'national review', 'nationalreview',
    'washington times', 'the washington times',
    'breitbart', 'breitbart news',
    # Common boilerplate patterns
    'associated press contributed', 'reporting by', 'editing by',
    'compiled by', 'written by',
]

def strip_source_identifiers(text):
    """Remove publication names and outlet-specific boilerplate from article text."""
    if not isinstance(text, str):
        return text
    cleaned = text
    for source in SOURCES_TO_STRIP:
        pattern = r'\b' + re.escape(source) + r'\b'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    # Collapse multiple spaces left by removals
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    return cleaned

# Load data
print("\nLoading dataset...")
df = pd.read_csv('news_articles_hf.csv')

print(f"Total articles: {len(df):,}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Strip source identifiers from article text
print("\nStripping source identifiers from article text...")
df['text'] = df['text'].apply(strip_source_identifiers)
print("  Done. Source names, bylines, and outlet boilerplate removed.")

# Encode labels
label2id = {'left': 0, 'center': 1, 'right': 2}
id2label = {0: 'left', 1: 'center', 2: 'right'}

df['labels'] = df['label'].map(label2id)

# Verify encoding worked
print("\nEncoded label distribution:")
print(df['labels'].value_counts().sort_index())

# Train/test split (stratified)
train_df, test_df = train_test_split(
    df,
    test_size=0.15,  # 85% train, 15% test
    stratify=df['labels'],
    random_state=42
)

print(f"\nTrain size: {len(train_df):,}")
print(f"Test size: {len(test_df):,}")

# Verify split is balanced
print("\nTrain label distribution:")
print(train_df['labels'].value_counts().sort_index())
print("\nTest label distribution:")
print(test_df['labels'].value_counts().sort_index())

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

# Load tokenizer
print("\nLoading DistilBERT tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding=False,  # Will pad dynamically in batches
        truncation=True,
        max_length=512
    )

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load model
print("\nLoading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

print(f"Model config check:")
print(f"  id2label: {model.config.id2label}")
print(f"  label2id: {model.config.label2id}")

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bias-classifier",
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
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,  # Only keep 2 best checkpoints
)

# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = (predictions == labels).mean()
    
    # Per-class accuracy
    per_class_acc = {}
    for label_id, label_name in id2label.items():
        mask = labels == label_id
        if mask.sum() > 0:
            per_class_acc[f'acc_{label_name}'] = (predictions[mask] == labels[mask]).mean()
    
    return {
        'accuracy': accuracy,
        **per_class_acc
    }

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Baseline evaluation (before training)
print("\n" + "="*60)
print("BASELINE (BEFORE TRAINING)")
print("="*60)
baseline = trainer.evaluate()
print("\nUntrained model performance:")
for metric, value in baseline.items():
    if metric.startswith('eval_'):
        print(f"  {metric}: {value:.4f}")

# Train
print("\n" + "="*60)
print("TRAINING")
print("="*60)
print("\nThis will take 1-3 hours depending on GPU...")
print("Progress will be logged every 100 steps")
print("You should see accuracy improve from ~33% to 85%+\n")

trainer.train()

# Evaluate
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

eval_results = trainer.evaluate()

print("\nTest set performance:")
for metric, value in eval_results.items():
    if metric.startswith('eval_'):
        print(f"  {metric}: {value:.4f}")

# Check if model learned properly
accuracy = eval_results['eval_accuracy']
acc_left = eval_results.get('eval_acc_left', 0)
acc_center = eval_results.get('eval_acc_center', 0)
acc_right = eval_results.get('eval_acc_right', 0)

print("\n" + "="*60)
print("TRAINING VERIFICATION")
print("="*60)

if accuracy < 0.60:
    print("⚠ WARNING: Accuracy is very low - model may not have learned")
    print("  Expected: 85%+")
    print(f"  Got: {accuracy*100:.1f}%")
elif abs(acc_left - acc_center) > 0.3 or abs(acc_left - acc_right) > 0.3:
    print("⚠ WARNING: Per-class accuracies are very imbalanced")
    print("  This suggests the model is biased toward one class")
else:
    print("✓ Training appears successful!")
    print(f"  Overall accuracy: {accuracy*100:.1f}%")
    print(f"  Per-class accuracies are balanced")

# Save model locally
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model_path = "./bias-classifier-final"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

print(f"\n✓ Model saved to: {model_path}")

# Verify saved model works
print("\nVerifying saved model...")
from transformers import pipeline

classifier = pipeline("text-classification", model=model_path, top_k=None)

test_texts = {
    "right": "The radical left socialist agenda is destroying America!",
    "center": "The Federal Reserve announced interest rates remain unchanged.",
    "left": "Corporate greed demands bold progressive action!"
}

print("\nQuick prediction test:")
all_correct = True
for expected, text in test_texts.items():
    result = classifier(text)[0]
    predicted = max(result, key=lambda x: x['score'])
    predicted_label = predicted['label'].replace('LABEL_', '')
    
    # Map LABEL_0/1/2 to left/center/right
    label_map = {'0': 'left', '1': 'center', '2': 'right'}
    if predicted_label in label_map:
        predicted_label = label_map[predicted_label]
    
    match = "✓" if predicted_label.lower() == expected else "✗"
    print(f"  {match} Expected {expected}, got {predicted_label} ({predicted['score']:.2f})")
    
    if predicted_label.lower() != expected:
        all_correct = False

if not all_correct:
    print("\n⚠ WARNING: Model predictions don't match expectations")
    print("  Model may not have trained correctly")
    print("  Check the metrics above before uploading")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)

if accuracy >= 0.60 and all_correct:
    print("\n✓ Model is ready to upload to Hugging Face!")
    print("\nRun: python upload_to_huggingface.py")
else:
    print("\n⚠ Model may have issues - review metrics above")
    print("  Consider retraining if results look wrong")

print("="*60)
