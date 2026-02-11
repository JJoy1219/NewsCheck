"""
Cross-source evaluation: Train on some outlets, test on others
This is the DEFINITIVE test for source leakage
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('news_articles_hf.csv')

print("="*60)
print("CROSS-SOURCE EVALUATION")
print("="*60)
print("\nThe definitive test for source leakage")
print("Train on outlets A, B, C → Test on outlet D")
print("\nIf performance holds, model learned content bias")
print("If performance tanks, model memorized sources")

# Check what publications we have
print(f"\n{'='*60}")
print("AVAILABLE PUBLICATIONS")
print(f"{'='*60}")
print(df.groupby(['label', 'publication']).size())

# Define train/test split BY SOURCE
# We'll train on some outlets and test on held-out outlets with same label

TRAIN_SOURCES = {
    'left': ['Vox', 'Vice News'],  # Train on these
    'center': ['Reuters', 'Business Insider'],
    'right': ['Fox News']  # Adjust based on what's available
}

TEST_SOURCES = {
    'left': ['Vice', 'Huffington Post', 'Buzzfeed News'],  # Test on these
    'center': ['NPR', 'BBC'],
    'right': ['New York Post', 'Breitbart']
}

# Check which sources are actually present
available_pubs = set(df['publication'].unique())

print(f"\n{'='*60}")
print("CHECKING TRAIN/TEST SOURCE AVAILABILITY")
print(f"{'='*60}")

# Verify and adjust splits based on what's actually in the dataset
train_mask = []
test_mask = []

for label in ['left', 'center', 'right']:
    train_pubs = [p for p in TRAIN_SOURCES.get(label, []) if p in available_pubs]
    test_pubs = [p for p in TEST_SOURCES.get(label, []) if p in available_pubs]
    
    print(f"\n{label.upper()}:")
    print(f"  Train sources: {train_pubs}")
    print(f"  Test sources: {test_pubs}")
    
    if not train_pubs:
        print(f"  ⚠️  No train sources available for {label}!")
    if not test_pubs:
        print(f"  ⚠️  No test sources available for {label}!")
    
    # Create masks
    train_mask.append(df['publication'].isin(train_pubs) & (df['label'] == label))
    test_mask.append(df['publication'].isin(test_pubs) & (df['label'] == label))

# Combine masks
train_mask_combined = pd.concat(train_mask, axis=1).any(axis=1)
test_mask_combined = pd.concat(test_mask, axis=1).any(axis=1)

df_train = df[train_mask_combined]
df_test = df[test_mask_combined]

print(f"\n{'='*60}")
print("TRAIN/TEST SPLIT SIZES")
print(f"{'='*60}")
print(f"\nTrain: {len(df_train)} articles")
print(df_train['label'].value_counts())
print(f"\nTest: {len(df_test)} articles")
print(df_test['label'].value_counts())

if len(df_train) == 0 or len(df_test) == 0:
    print("\n❌ ERROR: Not enough publications for cross-source eval")
    print("Need at least 2 publications per label")
    print("\nAvailable publications:")
    print(df.groupby('label')['publication'].unique())
    exit(1)

# Check for class imbalance
train_counts = df_train['label'].value_counts()
test_counts = df_test['label'].value_counts()

if train_counts.min() < 50:
    print(f"\n⚠️  WARNING: Small train set ({train_counts.min()} for {train_counts.idxmin()})")
    print("Results may be unreliable")

if test_counts.min() < 20:
    print(f"\n⚠️  WARNING: Small test set ({test_counts.min()} for {test_counts.idxmin()})")
    print("Results may be unreliable")

# Train model
print(f"\n{'='*60}")
print("TRAINING MODEL (on train sources only)")
print(f"{'='*60}")

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train = vectorizer.fit_transform(df_train['text'])
y_train = df_train['label']

X_test = vectorizer.transform(df_test['text'])
y_test = df_test['label']

clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

clf.fit(X_train, y_train)

# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print(f"\nTrain accuracy (on training sources): {train_acc:.1%}")
print(f"Test accuracy (on UNSEEN sources):    {test_acc:.1%}")

# Make predictions
y_pred = clf.predict(X_test)

print(f"\n{'='*60}")
print("PERFORMANCE ON UNSEEN SOURCES")
print(f"{'='*60}")
print(classification_report(y_test, y_pred))

# Compare to random baseline
random_baseline = 1 / len(df['label'].unique())
print(f"\n{'='*60}")
print("RESULTS INTERPRETATION")
print(f"{'='*60}")

print(f"\nRandom baseline:  {random_baseline:.1%}")
print(f"Cross-source acc: {test_acc:.1%}")
print(f"Improvement:      {(test_acc - random_baseline):.1%}")

if test_acc < 0.45:
    print(f"\n❌ SEVERE SOURCE LEAKAGE")
    print(f"   Model fails on unseen sources (< 45% accuracy)")
    print(f"   It memorized outlet names, not content bias")
    print(f"\n   The 89% accuracy on mixed-source eval was fake!")
elif test_acc < 0.55:
    print(f"\n⚠️  MODERATE SOURCE LEAKAGE")
    print(f"   Model struggles on unseen sources (45-55% accuracy)")
    print(f"   Some content learning, but still source-dependent")
elif test_acc < 0.65:
    print(f"\n✓ GOOD: Some generalization")
    print(f"   Model works okay on unseen sources (55-65%)")
    print(f"   Learning genuine bias patterns")
else:
    print(f"\n✓✓ EXCELLENT: Strong generalization")
    print(f"   Model performs well on unseen sources (65%+)")
    print(f"   Robustly learning content-based bias")

# Show per-source performance
print(f"\n{'='*60}")
print("PER-SOURCE TEST PERFORMANCE")
print(f"{'='*60}")

for pub in df_test['publication'].unique():
    pub_mask = df_test['publication'] == pub
    pub_acc = accuracy_score(
        df_test[pub_mask]['label'],
        clf.predict(vectorizer.transform(df_test[pub_mask]['text']))
    )
    pub_count = pub_mask.sum()
    print(f"{pub:25s} ({pub_count:4d} articles): {pub_acc:.1%}")

print(f"\n{'='*60}")
print("NEXT STEPS")
print(f"{'='*60}")

if test_acc >= 0.60:
    print("\n✓ Model generalizes across sources!")
    print("\nRecommended next steps:")
    print("  1. Scale up data (train on 5k per label)")
    print("  2. Try transformer model (DistilBERT)")
    print("  3. Test temporal robustness (train 2016-2018, test 2019)")
else:
    print("\n⚠️  Model struggles with unseen sources")
    print("\nRecommended next steps:")
    print("  1. Add more diverse sources to training")
    print("  2. Use data augmentation (paraphrasing)")
    print("  3. Try domain adaptation techniques")
    print("  4. Consider transformer model (better at generalizing)")

print(f"\n{'='*60}")
