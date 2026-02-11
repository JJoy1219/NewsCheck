"""
Test for source leakage by masking publication names
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('news_articles_hf.csv')

print("="*60)
print("SOURCE LEAKAGE TEST")
print("="*60)
print("\nTesting if model relies on source names...")

# List of publication names and common variations to mask
SOURCES_TO_MASK = [
    'reuters', 'associated press', 'ap news',
    'cnn', 'vox', 'vice', 'vice news',
    'fox news', 'fox', 'new york post', 'ny post', 'nypost',
    'new york times', 'nyt', 'times',
    'huffington post', 'huffpost', 'huff post',
    'buzzfeed', 'buzzfeed news',
    'guardian', 'the guardian',
    'washington post', 'wapo',
    'npr', 'bbc',
    'breitbart', 'national review'
]

def mask_sources(text):
    """Remove source names from text"""
    text_lower = text.lower()
    for source in SOURCES_TO_MASK:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(source) + r'\b'
        text_lower = re.sub(pattern, '[SOURCE]', text_lower, flags=re.IGNORECASE)
    return text_lower

# Create masked version
df['text_masked'] = df['text'].apply(mask_sources)

print(f"\nExample of masking:")
print(f"\nOriginal text preview:")
sample_text = df.iloc[0]['text'][:300]
print(sample_text)
print(f"\nMasked text preview:")
print(df.iloc[0]['text_masked'][:300])

# Split data
X_original = df['text']
X_masked = df['text_masked']
y = df['label']

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, stratify=y, random_state=42
)

X_train_mask, X_test_mask, _, _ = train_test_split(
    X_masked, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n{'='*60}")
print("BASELINE: Training on ORIGINAL text (with source names)")
print(f"{'='*60}")

# Train on original
vectorizer_orig = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
X_train_tfidf_orig = vectorizer_orig.fit_transform(X_train_orig)
X_test_tfidf_orig = vectorizer_orig.transform(X_test_orig)

clf_orig = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf_orig.fit(X_train_tfidf_orig, y_train)

acc_orig = clf_orig.score(X_test_tfidf_orig, y_test)
print(f"\nTest accuracy (with source names): {acc_orig:.1%}")

# Check for source features
feature_names_orig = vectorizer_orig.get_feature_names_out()
source_features_orig = [f for f in feature_names_orig 
                       if any(source in f.lower() for source in SOURCES_TO_MASK)]
print(f"Source-related features found: {len(source_features_orig)}")
if source_features_orig[:10]:
    print(f"Examples: {source_features_orig[:10]}")

print(f"\n{'='*60}")
print("TEST: Training on MASKED text (source names removed)")
print(f"{'='*60}")

# Train on masked
vectorizer_mask = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
X_train_tfidf_mask = vectorizer_mask.fit_transform(X_train_mask)
X_test_tfidf_mask = vectorizer_mask.transform(X_test_mask)

clf_mask = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf_mask.fit(X_train_tfidf_mask, y_train)

acc_mask = clf_mask.score(X_test_tfidf_mask, y_test)
print(f"\nTest accuracy (masked source names): {acc_mask:.1%}")

# Check for source features
feature_names_mask = vectorizer_mask.get_feature_names_out()
source_features_mask = [f for f in feature_names_mask 
                       if any(source in f.lower() for source in SOURCES_TO_MASK)]
print(f"Source-related features found: {len(source_features_mask)}")
if source_features_mask:
    print(f"Examples: {source_features_mask[:10]}")

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")

accuracy_drop = acc_orig - acc_mask
print(f"\nAccuracy with source names:    {acc_orig:.1%}")
print(f"Accuracy without source names: {acc_mask:.1%}")
print(f"Accuracy drop:                 {accuracy_drop:.1%}")

if accuracy_drop > 0.10:
    print(f"\n⚠️  SEVERE SOURCE LEAKAGE DETECTED")
    print(f"   Model relies heavily on source names")
    print(f"   Accuracy dropped by {accuracy_drop:.1%} when sources masked")
    print(f"\n   This model is NOT learning content bias!")
    print(f"   It's just memorizing outlet names.")
elif accuracy_drop > 0.05:
    print(f"\n⚠️  MODERATE SOURCE LEAKAGE")
    print(f"   Model partially relies on source names")
    print(f"   Accuracy dropped by {accuracy_drop:.1%}")
else:
    print(f"\n✓ MINIMAL SOURCE LEAKAGE")
    print(f"   Model appears to learn from content")
    print(f"   Accuracy only dropped by {accuracy_drop:.1%}")

# Show top features for masked model
print(f"\n{'='*60}")
print("TOP FEATURES (MASKED MODEL)")
print(f"{'='*60}")

for i, label in enumerate(clf_mask.classes_):
    coef = clf_mask.coef_[i]
    top_indices = np.argsort(coef)[-10:][::-1]
    top_features = [feature_names_mask[idx] for idx in top_indices]
    
    print(f"\n{label.upper()}:")
    for feat in top_features:
        print(f"  {feat}")

# Classification report
print(f"\n{'='*60}")
print("MASKED MODEL PERFORMANCE BY CLASS")
print(f"{'='*60}")
print(classification_report(y_test, clf_mask.predict(X_test_tfidf_mask)))

print(f"\n{'='*60}")
print("RECOMMENDATION")
print(f"{'='*60}")

if acc_mask < 0.55:
    print("\n⚠️  Masked accuracy < 55%")
    print("   Task may be too difficult without source names")
    print("   OR labels are too noisy")
    print("\n   Next steps:")
    print("   1. Examine misclassified articles")
    print("   2. Check if label assignments are correct")
    print("   3. Try transformer model (BERT) for better features")
elif acc_mask >= 0.60:
    print("\n✓ Masked accuracy ≥ 60%")
    print("   Model is learning genuine content patterns!")
    print("\n   Next steps:")
    print("   1. Run cross-source evaluation")
    print("   2. Try transformer fine-tuning")
    print("   3. Scale up to more data (5k per label)")
else:
    print("\n⚠️  Borderline performance (55-60%)")
    print("   Some content learning, but weak")
    print("\n   Next steps:")
    print("   1. Add more data")
    print("   2. Try feature engineering (sentiment, formality)")
    print("   3. Consider transformer model")

print(f"\n{'='*60}")
