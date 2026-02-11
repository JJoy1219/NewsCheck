import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('news_articles_hf.csv')

print("="*60)
print("BIAS CLASSIFICATION BASELINE")
print("="*60)
print(f"\nDataset size: {len(df)} articles")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Check text lengths
df['text_length'] = df['text'].str.len()
print(f"\nText statistics:")
print(f"  Mean length: {df['text_length'].mean():.0f} chars")
print(f"  Min length: {df['text_length'].min():.0f} chars")
print(f"  Max length: {df['text_length'].max():.0f} chars")

# Split data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# TF-IDF Vectorization
print("\n" + "="*60)
print("FEATURE EXTRACTION")
print("="*60)

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # unigrams and bigrams
    min_df=1,
    max_df=0.9,
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Train classifier
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION")
print("="*60)

clf = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

clf.fit(X_train_tfidf, y_train)

# Evaluate
train_acc = clf.score(X_train_tfidf, y_train)
test_acc = clf.score(X_test_tfidf, y_test)

print(f"\nTrain accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# Cross-validation (if enough data)
if len(X_train) >= 10:
    cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=min(3, len(X_train)), scoring='accuracy')
    print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Predictions
y_pred = clf.predict(X_test_tfidf)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred, labels=['left', 'center', 'right'])
print("\nPredicted →")
print("           left  center  right")
for i, label in enumerate(['left', 'center', 'right']):
    print(f"{label:>6}     {cm[i][0]:>4}  {cm[i][1]:>6}  {cm[i][2]:>5}")

# Top features per class
print("\n" + "="*60)
print("TOP PREDICTIVE FEATURES")
print("="*60)

feature_names = vectorizer.get_feature_names_out()
n_top = 10

for i, label in enumerate(clf.classes_):
    # Get coefficients for this class (one-vs-rest)
    coef = clf.coef_[i]
    top_indices = np.argsort(coef)[-n_top:][::-1]
    top_features = [feature_names[idx] for idx in top_indices]
    top_weights = [coef[idx] for idx in top_indices]
    
    print(f"\n{label.upper()}:")
    for feat, weight in zip(top_features, top_weights):
        print(f"  {feat:25s} {weight:6.3f}")

# Test for source leakage
print("\n" + "="*60)
print("SOURCE LEAKAGE CHECK")
print("="*60)

# Check if model learned source names
source_features = ['cnn', 'fox', 'reuters', 'nyt', 'times', 'news']
found_sources = [f for f in feature_names if any(s in f.lower() for s in source_features)]

if found_sources:
    print("WARNING: Source-related features detected in vocabulary:")
    for feat in found_sources[:5]:
        print(f"  - {feat}")
    print("\nModel may be learning source artifacts rather than content bias.")
else:
    print("No obvious source-related features in top vocabulary.")

# Test predictions on sample articles
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

for idx in X_test.index[:3]:
    article = df.loc[idx]
    text_vec = vectorizer.transform([article['text']])
    pred = clf.predict(text_vec)[0]
    proba = clf.predict_proba(text_vec)[0]
    
    print(f"\nTitle: {article['title'][:70]}")
    print(f"True: {article['label']:8s} | Pred: {pred:8s}")
    print(f"Confidence: {dict(zip(clf.classes_, proba))}")

# Save model summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Test accuracy: {test_acc:.1%}")
print(f"Dataset size: {len(df)} articles")
print(f"Random baseline: {1/len(clf.classes_):.1%}")
print(f"Improvement over random: {(test_acc - 1/len(clf.classes_)):.1%}")

if test_acc > 0.8:
    print("\n⚠️  High accuracy - check for:")
    print("  - Source leakage (model memorizing outlet names)")
    print("  - Dataset too small / overfitting")
    print("  - Task too easy (unrealistic bias signals)")
elif test_acc < 0.4:
    print("\n⚠️  Low accuracy - possible issues:")
    print("  - Dataset too small")
    print("  - Labels too noisy")
    print("  - Task genuinely difficult")
else:
    print("\n✓ Reasonable performance - task appears learnable")
    print("  Next steps:")
    print("  - Collect more data (target 500+ articles)")
    print("  - Test transformer model (BERT/RoBERTa)")
    print("  - Cross-source evaluation")
