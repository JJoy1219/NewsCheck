"""
Temporal Robustness Test: Train on earlier articles, test on later articles.
This confirms whether temporal distribution shift is degrading predictions.

If accuracy drops significantly on later articles, the model is learning
time-specific patterns (topics, entities, events) rather than stable bias signals.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
df = pd.read_csv('news_articles_hf.csv')

print("=" * 60)
print("TEMPORAL ROBUSTNESS TEST")
print("=" * 60)
print("\nTrain on earlier articles -> Test on later articles")
print("If accuracy holds, model learned stable bias signals")
print("If accuracy drops, model memorized time-specific patterns")

# Parse dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total articles with valid dates: {len(df):,}")

# Show year distribution
df['year'] = df['date'].dt.year
print(f"\nArticles per year:")
print(df['year'].value_counts().sort_index())

print(f"\nArticles per year per label:")
print(df.groupby(['year', 'label']).size().unstack(fill_value=0))

# ============================================================
# Split by time: train on earlier half, test on later half
# ============================================================
median_date = df['date'].median()
print(f"\nMedian date (split point): {median_date.date()}")

df_early = df[df['date'] < median_date].copy()
df_late = df[df['date'] >= median_date].copy()

print(f"\nEarly period: {df_early['date'].min().date()} to {df_early['date'].max().date()}")
print(f"  Articles: {len(df_early):,}")
print(f"  Label distribution:")
print(f"  {df_early['label'].value_counts().to_dict()}")

print(f"\nLate period: {df_late['date'].min().date()} to {df_late['date'].max().date()}")
print(f"  Articles: {len(df_late):,}")
print(f"  Label distribution:")
print(f"  {df_late['label'].value_counts().to_dict()}")

# ============================================================
# Baseline: Random split (time-mixed) for comparison
# ============================================================
print(f"\n{'=' * 60}")
print("BASELINE: RANDOM SPLIT (time-mixed)")
print(f"{'=' * 60}")

from sklearn.model_selection import train_test_split

X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(
    df['text'], df['label'], test_size=0.3, stratify=df['label'], random_state=42
)

vectorizer_rand = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_rand_train_tfidf = vectorizer_rand.fit_transform(X_rand_train)
X_rand_test_tfidf = vectorizer_rand.transform(X_rand_test)

clf_rand = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf_rand.fit(X_rand_train_tfidf, y_rand_train)

rand_acc = clf_rand.score(X_rand_test_tfidf, y_rand_test)
print(f"\nRandom-split accuracy: {rand_acc:.1%}")
print(classification_report(y_rand_test, clf_rand.predict(X_rand_test_tfidf)))

# ============================================================
# Temporal split: train early, test late
# ============================================================
print(f"\n{'=' * 60}")
print("TEMPORAL SPLIT: Train early -> Test late")
print(f"{'=' * 60}")

vectorizer_temp = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_early_tfidf = vectorizer_temp.fit_transform(df_early['text'])
X_late_tfidf = vectorizer_temp.transform(df_late['text'])

clf_temp = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf_temp.fit(X_early_tfidf, df_early['label'])

temp_train_acc = clf_temp.score(X_early_tfidf, df_early['label'])
temp_test_acc = clf_temp.score(X_late_tfidf, df_late['label'])

print(f"\nTrain accuracy (early period): {temp_train_acc:.1%}")
print(f"Test accuracy (late period):   {temp_test_acc:.1%}")

y_temp_pred = clf_temp.predict(X_late_tfidf)
print(f"\nClassification report (late period):")
print(classification_report(df_late['label'], y_temp_pred))

# Confusion matrix
cm = confusion_matrix(df_late['label'], y_temp_pred, labels=['left', 'center', 'right'])
print("Confusion matrix (late period):")
print("Predicted ->     left  center  right")
for i, label in enumerate(['left', 'center', 'right']):
    print(f"  {label:>6}     {cm[i][0]:>5}  {cm[i][1]:>6}  {cm[i][2]:>5}")

# ============================================================
# Prediction distribution analysis
# ============================================================
print(f"\n{'=' * 60}")
print("PREDICTION DISTRIBUTION (late period)")
print(f"{'=' * 60}")

pred_dist = pd.Series(y_temp_pred).value_counts()
true_dist = df_late['label'].value_counts()

print(f"\n{'Label':<10} {'True Count':>12} {'Predicted Count':>16} {'Ratio':>8}")
print("-" * 50)
for label in ['left', 'center', 'right']:
    true_count = true_dist.get(label, 0)
    pred_count = pred_dist.get(label, 0)
    ratio = pred_count / true_count if true_count > 0 else 0
    print(f"{label:<10} {true_count:>12,} {pred_count:>16,} {ratio:>8.2f}")

# ============================================================
# Per-year accuracy breakdown
# ============================================================
print(f"\n{'=' * 60}")
print("PER-YEAR ACCURACY")
print(f"{'=' * 60}")

for year in sorted(df['year'].unique()):
    year_mask = df_late['year'] == year
    if year_mask.sum() > 0:
        year_acc = accuracy_score(
            df_late[year_mask]['label'],
            clf_temp.predict(vectorizer_temp.transform(df_late[year_mask]['text']))
        )
        year_preds = clf_temp.predict(vectorizer_temp.transform(df_late[year_mask]['text']))
        year_pred_dist = pd.Series(year_preds).value_counts().to_dict()
        print(f"  {year}: {year_acc:.1%} accuracy ({year_mask.sum():,} articles) | pred dist: {year_pred_dist}")

# ============================================================
# Top temporal features analysis
# ============================================================
print(f"\n{'=' * 60}")
print("TOP FEATURES (temporal model)")
print(f"{'=' * 60}")

feature_names = vectorizer_temp.get_feature_names_out()

for i, label in enumerate(clf_temp.classes_):
    coef = clf_temp.coef_[i]
    top_indices = np.argsort(coef)[-15:][::-1]
    top_features = [(feature_names[idx], coef[idx]) for idx in top_indices]

    print(f"\n{label.upper()}:")
    for feat, weight in top_features:
        print(f"  {feat:30s} {weight:7.3f}")

# ============================================================
# Results interpretation
# ============================================================
print(f"\n{'=' * 60}")
print("RESULTS")
print(f"{'=' * 60}")

accuracy_drop = rand_acc - temp_test_acc
print(f"\nRandom-split accuracy:   {rand_acc:.1%}")
print(f"Temporal-split accuracy: {temp_test_acc:.1%}")
print(f"Accuracy drop:           {accuracy_drop:.1%}")

if accuracy_drop > 0.15:
    print(f"\n[X] SEVERE TEMPORAL DEGRADATION ({accuracy_drop:.1%} drop)")
    print("   The model heavily relies on time-specific patterns.")
    print("   Training on 2016-2020 data will NOT generalize to 2024+ articles.")
    print("   This confirms temporal shift as a primary cause of the")
    print("   'always predicts left' behavior on modern articles.")
elif accuracy_drop > 0.08:
    print(f"\n[!] MODERATE TEMPORAL DEGRADATION ({accuracy_drop:.1%} drop)")
    print("   The model partially relies on time-specific patterns.")
    print("   Modern articles will see reduced accuracy.")
elif accuracy_drop > 0.03:
    print(f"\n[!] MILD TEMPORAL DEGRADATION ({accuracy_drop:.1%} drop)")
    print("   Some temporal sensitivity, but model is fairly robust.")
else:
    print(f"\n[OK] TEMPORALLY ROBUST ({accuracy_drop:.1%} drop)")
    print("   Model generalizes well across time periods.")

# Check if predictions skew toward one class
max_pred_pct = pred_dist.max() / pred_dist.sum()
dominant_class = pred_dist.idxmax()
if max_pred_pct > 0.45:
    print(f"\n[!] PREDICTION SKEW: {max_pred_pct:.0%} of predictions are '{dominant_class}'")
    print(f"   This mirrors the 'always predicts left' behavior seen on modern articles.")

print(f"\n{'=' * 60}")
