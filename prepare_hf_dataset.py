"""
Load and prepare All The News 2.1 dataset from Hugging Face
Maps publications to bias labels and filters for political content
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

print("Loading dataset from Hugging Face...")
print("(This may take a few minutes - 2.69M articles)")

# Load the dataset
dataset = load_dataset("rjac/all-the-news-2-1-Component-one")

# Convert to pandas for easier manipulation
df = pd.DataFrame(dataset['train'])

print(f"\nLoaded {len(df):,} articles")
print(f"\nPublications in dataset:")
print(df['publication'].value_counts())

# Map publications to bias labels based on AllSides Media Bias Ratings
# AllSides rates outlets as: Left, Lean Left, Center, Lean Right, Right

BIAS_MAP = {
    # LEFT
    'Vox': 'left',
    'Vice': 'left',
    'Vice News': 'left',
    'Huffington Post': 'left',
    'Buzzfeed News': 'left',
    'Guardian': 'left',
    'New York Times': 'left',
    
    # CENTER
    'Reuters': 'center',
    'Associated Press': 'center',
    'AP': 'center',
    'BBC': 'center',
    'Business Insider': 'center',
    'The Hill': 'center',
    'NPR': 'center',
    'USA Today': 'center',
    
    # RIGHT
    'Fox News': 'right',
    'New York Post': 'right',
    'National Review': 'right',
    'Washington Times': 'right',
    'Breitbart': 'right',
    
    # Exclude non-political
    'TMZ': None,
    'Hyperallergic': None,
    'People': None,
    'Entertainment Weekly': None,
}

# Apply bias labels
df['label'] = df['publication'].map(BIAS_MAP)

# Remove articles without bias labels (non-political outlets)
df_political = df[df['label'].notna()].copy()

print(f"\n{'='*60}")
print("AFTER FILTERING FOR POLITICAL OUTLETS")
print(f"{'='*60}")
print(f"Total political articles: {len(df_political):,}")
print(f"\nLabel distribution:")
print(df_political['label'].value_counts())

# Filter for articles with substantial content
print(f"\nFiltering for articles with >200 characters...")
df_political = df_political[df_political['article'].str.len() > 200]

print(f"After length filter: {len(df_political):,} articles")
print(f"\nLabel distribution:")
print(df_political['label'].value_counts())

# Sample balanced dataset
# Get minimum count across labels
min_count = df_political['label'].value_counts().min()

# Sample equally from each label
print(f"\n{'='*60}")
print("CREATING BALANCED DATASET")
print(f"{'='*60}")

# Target: 10,000 articles per label (30,000 total) for production model
target_per_label = min(10000, min_count)

df_balanced = pd.concat([
    df_political[df_political['label'] == 'left'].sample(target_per_label, random_state=42),
    df_political[df_political['label'] == 'center'].sample(target_per_label, random_state=42),
    df_political[df_political['label'] == 'right'].sample(target_per_label, random_state=42)
])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset size: {len(df_balanced):,} articles")
print(f"\nLabel distribution:")
print(df_balanced['label'].value_counts())
print(f"\nPublication distribution:")
print(df_balanced.groupby(['publication', 'label']).size())

# Create final dataset with just the columns we need
df_final = df_balanced[['title', 'article', 'publication', 'label', 'date', 'url']].copy()
df_final = df_final.rename(columns={'article': 'text'})

# Save to CSV
output_path = 'news_articles_hf.csv'
df_final.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print("DATASET STATISTICS")
print(f"{'='*60}")

# Text length statistics
df_final['text_length'] = df_final['text'].str.len()
df_final['word_count'] = df_final['text'].str.split().str.len()

print(f"\nText length (characters):")
print(f"  Mean: {df_final['text_length'].mean():.0f}")
print(f"  Median: {df_final['text_length'].median():.0f}")
print(f"  Min: {df_final['text_length'].min():.0f}")
print(f"  Max: {df_final['text_length'].max():.0f}")

print(f"\nWord count:")
print(f"  Mean: {df_final['word_count'].mean():.0f}")
print(f"  Median: {df_final['word_count'].median():.0f}")

# Sample articles
print(f"\n{'='*60}")
print("SAMPLE ARTICLES")
print(f"{'='*60}")

for label in ['left', 'center', 'right']:
    sample = df_final[df_final['label'] == label].iloc[0]
    print(f"\n{label.upper()} - {sample['publication']}")
    print(f"Title: {sample['title'][:80]}...")
    print(f"Text preview: {sample['text'][:200]}...")

print(f"\n{'='*60}")
print(f"✓ Saved balanced dataset to: {output_path}")
print(f"✓ {len(df_final):,} articles ready for training")
print(f"\nNext steps:")
print(f"  1. Run baseline_classifier.py on this dataset")
print(f"  2. Check cross-source generalization")
print(f"  3. Consider expanding to 1500-2000 per label if results look good")
print(f"{'='*60}")
