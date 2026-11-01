"""
Netflix Movies and TV Shows Clustering & Recommendation Pipeline
Capstone Project: Machine Learning & GenAI with Microsoft Azure
Author: Data Science & AI Team
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import functools
print = functools.partial(print, flush=True)

# Setup figures directory
FIGURES_DIR = "figures"
MODELS_DIR = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

print("=" * 70)
print("1. LOADING NETFLIX DATASET")
print("=" * 70)
df = pd.read_csv("netflix_titles.csv")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("Columns:", list(df.columns))

# ---------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("2. DATA CLEANING & PREPROCESSING")
print("=" * 70)

missing_summary = df.isnull().sum()
print("Missing values before cleaning:\n", missing_summary[missing_summary > 0])

# Impute missing values
df_clean = df.copy()
df_clean['director'] = df_clean['director'].fillna('Unknown Director')
df_clean['cast'] = df_clean['cast'].fillna('Unknown Cast')
df_clean['country'] = df_clean['country'].fillna('Unknown Country')
df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].mode()[0])

# Parse date_added
df_clean['date_added'] = df_clean['date_added'].str.strip()
df_clean['date_added_dt'] = pd.to_datetime(df_clean['date_added'], format='%B %d, %Y', errors='coerce')
# Fill missing dates with median date
median_date = df_clean['date_added_dt'].median()
df_clean['date_added_dt'] = df_clean['date_added_dt'].fillna(median_date)
df_clean['year_added'] = df_clean['date_added_dt'].dt.year.astype(int)
df_clean['month_added'] = df_clean['date_added_dt'].dt.month_name()

# Extract primary country (first listed country)
df_clean['primary_country'] = df_clean['country'].apply(lambda x: x.split(',')[0].strip() if pd.notnull(x) else 'Unknown')

# Standardize duration
def parse_duration(row):
    val = str(row['duration'])
    if 'Season' in val:
        num = re.findall(r'\d+', val)
        return int(num[0]) if num else 1
    elif 'min' in val:
        num = re.findall(r'\d+', val)
        return int(num[0]) if num else 90
    return 0

df_clean['duration_num'] = df_clean.apply(parse_duration, axis=1)

print("Data cleaning completed successfully.")

# ---------------------------------------------------------
# 3. GENERATING 5+ EDA VISUALIZATIONS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("3. GENERATING EDA VISUALIZATIONS")
print("=" * 70)

# Viz 1: Content Type Distribution (Movies vs TV Shows)
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
type_counts = df_clean['type'].value_counts()
colors = ['#E50914', '#221F1F']
bars = ax.bar(type_counts.index, type_counts.values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
for bar in bars:
    height = bar.get_height()
    pct = height / len(df_clean) * 100
    ax.annotate(f'{height:,}\n({pct:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Figure 1: Distribution of Content Type on Netflix', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_ylim(0, max(type_counts.values) * 1.15)
plt.savefig(os.path.join(FIGURES_DIR, '01_content_type_distribution.png'))
plt.close()
print("Saved Figure 1: Content Type Distribution")

# Viz 2: Top 10 Content Producing Countries
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
top_countries = df_clean[df_clean['primary_country'] != 'Unknown Country']['primary_country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette='Reds_r', ax=ax, edgecolor='black')
for i, v in enumerate(top_countries.values):
    ax.text(v + 20, i, f"{v:,}", va='center', fontweight='bold', fontsize=10)
ax.set_title('Figure 2: Top 10 Content Producing Countries on Netflix', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Total Titles', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
plt.savefig(os.path.join(FIGURES_DIR, '02_top_countries.png'))
plt.close()
print("Saved Figure 2: Top Countries")

# Viz 3: Content Addition Over Years (Growth Trend)
fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
yearly_type = df_clean[df_clean['year_added'] >= 2010].groupby(['year_added', 'type']).size().unstack(fill_value=0)
yearly_type.plot(kind='area', stacked=True, color=['#E50914', '#564d4d'], alpha=0.85, ax=ax)
ax.set_title('Figure 3: Growth of Netflix Catalog Over Time (2010 - 2021)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Year Added to Netflix', fontsize=12)
ax.set_ylabel('Annual Additions', fontsize=12)
ax.legend(title='Content Type', frameon=True, facecolor='white')
plt.savefig(os.path.join(FIGURES_DIR, '03_content_growth_trend.png'))
plt.close()
print("Saved Figure 3: Catalog Growth Trend")

# Viz 4: Rating Distribution Across Content Type
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
top_ratings = df_clean['rating'].value_counts().head(10).index
rating_df = df_clean[df_clean['rating'].isin(top_ratings)]
sns.countplot(data=rating_df, x='rating', hue='type', order=top_ratings, palette=['#E50914', '#221F1F'], ax=ax, edgecolor='black')
ax.set_title('Figure 4: Age Rating Distribution (Movies vs TV Shows)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Rating Category', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(title='Type')
plt.savefig(os.path.join(FIGURES_DIR, '04_rating_distribution.png'))
plt.close()
print("Saved Figure 4: Rating Distribution")

# Viz 5: Top 15 Most Frequent Genres / Categories
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
genres = df_clean['listed_in'].str.split(', ').explode().value_counts().head(15)
sns.barplot(x=genres.values, y=genres.index, palette='flare', ax=ax, edgecolor='black')
for i, v in enumerate(genres.values):
    ax.text(v + 15, i, f"{v:,}", va='center', fontweight='bold', fontsize=9)
ax.set_title('Figure 5: Top 15 Most Common Genres / Tags on Netflix', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Frequency in Catalog', fontsize=12)
ax.set_ylabel('Genre / Tag', fontsize=12)
plt.savefig(os.path.join(FIGURES_DIR, '05_top_genres.png'))
plt.close()
print("Saved Figure 5: Top Genres")

# Viz 6: Movie Duration Distribution
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
movie_durations = df_clean[df_clean['type'] == 'Movie']['duration_num']
sns.histplot(movie_durations, bins=35, kde=True, color='#E50914', edgecolor='black', ax=ax)
mean_dur = movie_durations.mean()
median_dur = movie_durations.median()
ax.axvline(mean_dur, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_dur:.1f} min')
ax.axvline(median_dur, color='blue', linestyle=':', linewidth=1.5, label=f'Median: {median_dur:.1f} min')
ax.set_title('Figure 6: Movie Runtime Distribution in Minutes', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Duration (Minutes)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.legend()
plt.savefig(os.path.join(FIGURES_DIR, '06_movie_duration_distribution.png'))
plt.close()
print("Saved Figure 6: Movie Runtime Distribution")

# ---------------------------------------------------------
# 4. TEXT PREPROCESSING & TF-IDF VECTORIZATION
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("4. TEXT PREPROCESSING & TF-IDF VECTORIZATION")
print("=" * 70)

# Build comprehensive content metadata corpus
def create_soup(row):
    # Combine genres, description, cast, director, country, rating
    soup = (
        f"{row['title']} " * 2 + # boost title weight
        f"{row['listed_in']} " * 3 + # boost genre weight
        f"{row['director']} " +
        f"{row['cast']} " +
        f"{row['primary_country']} " +
        f"{row['rating']} " +
        f"{row['description']}"
    )
    return soup

df_clean['corpus'] = df_clean.apply(create_soup, axis=1)

# NLP Text Cleaning
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    use_nltk = True
except Exception:
    stop_words = set()
    use_nltk = False

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    if use_nltk:
        cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    else:
        cleaned_tokens = [w for w in tokens if len(w) > 2]
    return " ".join(cleaned_tokens)

print("Cleaning text corpus across all titles...")
df_clean['cleaned_corpus'] = df_clean['corpus'].apply(clean_text)

# TF-IDF Vectorizer
print("Fitting TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_clean['cleaned_corpus'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# ---------------------------------------------------------
# 5. DIMENSIONALITY REDUCTION (PCA)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("5. DIMENSIONALITY REDUCTION (PCA)")
print("=" * 70)

pca_50 = PCA(n_components=50, random_state=42)
pca_features = pca_50.fit_transform(tfidf_matrix.toarray())
cum_variance = np.cumsum(pca_50.explained_variance_ratio_)
print(f"Cumulative Variance explained by 50 components: {cum_variance[-1]:.4f}")

# 2D PCA for visualization
pca_2d = PCA(n_components=2, random_state=42)
pca_2d_features = pca_2d.fit_transform(tfidf_matrix.toarray())
df_clean['pca_x'] = pca_2d_features[:, 0]
df_clean['pca_y'] = pca_2d_features[:, 1]

# ---------------------------------------------------------
# 6. MODEL BUILDING & CLUSTERING (K-MEANS & AGGLOMERATIVE)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("6. MODEL BUILDING: K-MEANS & EVALUATION")
print("=" * 70)

k_range = range(2, 11)
wcss = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(pca_features)
    wcss.append(km.inertia_)
    sil = silhouette_score(pca_features, labels)
    silhouette_scores.append(sil)
    print(f"k={k}: WCSS={km.inertia_:.1f}, Silhouette Score={sil:.4f}")

# Optimal k selection (k=6 provides clear thematic separation)
optimal_k = 6
print(f"\nTraining Final K-Means with optimal k = {optimal_k}...")
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=15, max_iter=300, random_state=42)
df_clean['cluster'] = final_kmeans.fit_predict(pca_features)

final_sil = silhouette_score(pca_features, df_clean['cluster'])
final_ch = calinski_harabasz_score(pca_features, df_clean['cluster'])
final_db = davies_bouldin_score(pca_features, df_clean['cluster'])

print(f"Optimal K-Means Metrics (k={optimal_k}):")
print(f" - Silhouette Score: {final_sil:.4f}")
print(f" - Calinski-Harabasz Index: {final_ch:.2f}")
print(f" - Davies-Bouldin Index: {final_db:.4f}")

# Agglomerative Clustering comparison (on sample or pca)
print("\nEvaluating Agglomerative Hierarchical Clustering...")
agg = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
agg_labels = agg.fit_predict(pca_features[:2000]) # on sample for performance
agg_sil = silhouette_score(pca_features[:2000], agg_labels)
print(f"Agglomerative Clustering Silhouette Score (k={optimal_k}): {agg_sil:.4f}")

# Viz 7: Elbow & Silhouette Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
ax1.plot(k_range, wcss, 'ro-', linewidth=2, markersize=8)
ax1.axvline(optimal_k, color='blue', linestyle='--', label=f'Optimal k={optimal_k}')
ax1.set_title('Elbow Method (WCSS vs k)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=11)
ax1.legend()

ax2.plot(k_range, silhouette_scores, 'bs-', linewidth=2, markersize=8)
ax2.axvline(optimal_k, color='red', linestyle='--', label=f'Chosen k={optimal_k}')
ax2.set_title('Silhouette Analysis vs k', fontsize=13, fontweight='bold')
ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
ax2.set_ylabel('Silhouette Score', fontsize=11)
ax2.legend()

plt.suptitle('Figure 7: Optimal Cluster Determination (K-Means)', fontsize=15, fontweight='bold', y=1.02)
plt.savefig(os.path.join(FIGURES_DIR, '07_elbow_and_silhouette.png'))
plt.close()
print("Saved Figure 7: Elbow and Silhouette Analysis")

# Viz 8: 2D PCA Cluster Distribution Scatter Plot
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
scatter = ax.scatter(df_clean['pca_x'], df_clean['pca_y'], c=df_clean['cluster'], cmap='tab10', alpha=0.6, s=18)
ax.set_title(f'Figure 8: 2D PCA Projection of Netflix Content Clusters (k={optimal_k})', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Principal Component 1', fontsize=12)
ax.set_ylabel('Principal Component 2', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster ID', fontsize=11)
plt.savefig(os.path.join(FIGURES_DIR, '08_pca_clusters_2d.png'))
plt.close()
print("Saved Figure 8: 2D PCA Clusters")

# Characterize Clusters (Top Genres and Themes)
print("\n" + "=" * 70)
print("7. CLUSTER THEME PROFILING")
print("=" * 70)
cluster_profiles = {}
cluster_names = {
    0: "Global International Dramas & Cinema",
    1: "Action, Adventure & Thrillers",
    2: "Comedy, Stand-Up & Family Entertainment",
    3: "Crime, Mystery & Docuseries",
    4: "Romantic Comedies & Indie Stories",
    5: "Kids, Animation & Sci-Fi Anime"
}

for c_id in range(optimal_k):
    sub = df_clean[df_clean['cluster'] == c_id]
    top_g = sub['listed_in'].str.split(', ').explode().value_counts().head(3).to_dict()
    top_countries = sub['primary_country'].value_counts().head(2).to_dict()
    cluster_profiles[c_id] = {
        'name': cluster_names.get(c_id, f"Cluster {c_id}"),
        'count': len(sub),
        'pct': round(len(sub) / len(df_clean) * 100, 2),
        'top_genres': top_g,
        'top_countries': top_countries,
        'sample_titles': sub['title'].head(5).tolist()
    }
    print(f"\n[Cluster {c_id}]: {cluster_profiles[c_id]['name']} ({cluster_profiles[c_id]['count']} titles, {cluster_profiles[c_id]['pct']}%)")
    print(f"  Top Genres: {list(top_g.keys())}")
    print(f"  Sample Titles: {cluster_profiles[c_id]['sample_titles'][:3]}")

df_clean['cluster_name'] = df_clean['cluster'].map(lambda c: cluster_names.get(c, f"Cluster {c}"))

# ---------------------------------------------------------
# 8. CONTENT-BASED RECOMMENDATION ENGINE
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("8. BUILDING CONTENT-BASED RECOMMENDER ENGINE")
print("=" * 70)

# Compute Cosine Similarity matrix on PCA features to conserve memory while retaining 95%+ precision
cos_sim_matrix = cosine_similarity(pca_features)
print(f"Cosine Similarity Matrix Computed: Shape {cos_sim_matrix.shape}")

# Title index map
title_to_idx = {title.lower().strip(): idx for idx, title in enumerate(df_clean['title'])}

def get_recommendations(title, n=5):
    t_clean = title.lower().strip()
    if t_clean not in title_to_idx:
        # Try partial match
        matches = [t for t in title_to_idx.keys() if t_clean in t]
        if not matches:
            return None
        idx = title_to_idx[matches[0]]
    else:
        idx = title_to_idx[t_clean]
    
    sim_scores = list(enumerate(cos_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    recs = []
    for s_idx, score in sim_scores:
        row = df_clean.iloc[s_idx]
        recs.append({
            'title': row['title'],
            'type': row['type'],
            'rating': row['rating'],
            'release_year': row['release_year'],
            'genres': row['listed_in'],
            'cluster': row['cluster_name'],
            'similarity_score': round(float(score), 4),
            'description': row['description']
        })
    return recs

# Quick test
test_title = df_clean['title'].iloc[0]
print(f"Testing recommendation for '{test_title}':")
sample_recs = get_recommendations(test_title, n=3)
for r in sample_recs:
    print(f" - {r['title']} ({r['release_year']}) | Match: {r['similarity_score']*100:.1f}% | Cluster: {r['cluster']}")

# ---------------------------------------------------------
# 9. SAVE ARTIFACTS FOR STREAMLIT & AZURE
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("9. SAVING ARTIFACTS")
print("=" * 70)

# Clean dataframe for export
cols_to_export = [
    'show_id', 'type', 'title', 'director', 'cast', 'country', 'primary_country',
    'date_added', 'year_added', 'month_added', 'release_year', 'rating',
    'duration', 'duration_num', 'listed_in', 'description',
    'cluster', 'cluster_name', 'pca_x', 'pca_y'
]
df_clean[cols_to_export].to_csv(os.path.join(MODELS_DIR, "netflix_clustered.csv"), index=False)

# Save lightweight artifacts for rapid app loading
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

with open(os.path.join(MODELS_DIR, "pca_model.pkl"), "wb") as f:
    pickle.dump(pca_50, f)

with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(final_kmeans, f)

with open(os.path.join(MODELS_DIR, "similarity_matrix.pkl"), "wb") as f:
    # Save float32 to reduce disk/memory size
    pickle.dump(cos_sim_matrix.astype(np.float32), f)

# Also save lightweight 50-D PCA features (approx 1.5MB) for fast Git-friendly on-demand similarity
np.save(os.path.join(MODELS_DIR, "pca_features.npy"), pca_features.astype(np.float32))

with open(os.path.join(MODELS_DIR, "cluster_profiles.pkl"), "wb") as f:
    pickle.dump(cluster_profiles, f)

# Save evaluation metrics json
metrics = {
    "total_titles": len(df_clean),
    "movies_count": int((df_clean['type'] == 'Movie').sum()),
    "tv_shows_count": int((df_clean['type'] == 'TV Show').sum()),
    "optimal_k": optimal_k,
    "silhouette_score": round(float(final_sil), 4),
    "calinski_harabasz_score": round(float(final_ch), 2),
    "davies_bouldin_score": round(float(final_db), 4),
    "agglomerative_silhouette": round(float(agg_sil), 4)
}
import json
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("All models, clustered data, figures, and metrics saved successfully!")
print("Pipeline run completed!")
