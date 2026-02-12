"""
Fine-tune DistilBERT on 30k articles for production deployment

This model will be uploaded to Hugging Face Hub and called from GitHub Pages
"""

import pandas as pd
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

print("="*60)
print("TRAINING PRODUCTION BIAS CLASSIFIER")
print("="*60)

# Load data
print("\nLoading dataset...")
df = pd.read_csv('news_articles_hf.csv')

print(f"Total articles: {len(df):,}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Encode labels
label2id = {'left': 0, 'center': 1, 'right': 2}
id2label = {0: 'left', 1: 'center', 2: 'right'}

df['labels'] = df['label'].map(label2id)

# Train/test split (stratified)
train_df, test_df = train_test_split(
    df,
    test_size=0.15,  # 85% train, 15% test
    stratify=df['labels'],
    random_state=42
)

print(f"\nTrain size: {len(train_df):,}")
print(f"Test size: {len(test_df):,}")

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
    push_to_hub=False,  # Set to True if you want to auto-upload
    logging_dir='./logs',
    logging_steps=100,
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

# Train
print("\n" + "="*60)
print("TRAINING")
print("="*60)
print("\nThis will take 1-3 hours depending on GPU...")
print("Progress will be logged every 100 steps\n")

trainer.train()

# Evaluate
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

eval_results = trainer.evaluate()

print("\nTest set performance:")
for metric, value in eval_results.items():
    print(f"  {metric}: {value:.4f}")

# Save model locally
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model_path = "./bias-classifier-final"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

print(f"\n✓ Model saved to: {model_path}")

print("\n" + "="*60)
print("NEXT STEPS: DEPLOY TO HUGGING FACE")
print("="*60)

print("""
To deploy this model for your GitHub Pages website:

1. Create Hugging Face account (if you don't have one):
   https://huggingface.co/join

2. Install Hugging Face CLI:
   pip install huggingface_hub
   huggingface-cli login

3. Upload model:
   huggingface-cli upload your-username/news-bias-classifier ./bias-classifier-final

4. Your model will be available at:
   https://huggingface.co/your-username/news-bias-classifier

5. You can then call it from your GitHub Pages site using the Inference API

See deployment_guide.md for full instructions.
""")

print("="*60)