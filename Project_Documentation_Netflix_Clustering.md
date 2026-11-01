# Capstone Project Documentation: Netflix Movies & TV Shows Clustering and Recommendation System

**Program:** Machine Learning & GenAI with Microsoft Azure  
**Project Track:** Netflix Movies and TV Shows Clustering (Unsupervised Learning & Generative AI)  
**Author / Student:** Data Science & AI Professional  
**Date:** October 2026  
**Status:** Completed & Ready for Submission  

---

## 1. Executive Summary & Business Context

### 1.1 Executive Summary
With over 230 million global paid memberships across 190 countries, Netflix's multi-billion dollar platform hinges on user retention, session duration, and discovery satisfaction. The platform hosts thousands of titles with diverse origins, formats, genres, and narrative structures. This Capstone Project builds an **End-to-End Unsupervised Machine Learning and Generative AI System** designed to:
1. Segment Netflix's extensive library of 7,787 titles into coherent thematic and stylistic clusters using natural language processing (NLP) and unsupervised learning algorithms (K-Means, Agglomerative Hierarchical Clustering).
2. Overcome "catalog bloat" and "decision fatigue" through a content-based recommendation system powered by high-dimensional Cosine Similarity across metadata embeddings.
3. Bridge algorithmic recommendations and subscriber trust through an integrated **GenAI Explanation Copilot** (leveraging Google Gemini / Azure OpenAI) that explains *why* specific titles are grouped together in natural, human language.
4. Provide an interactive, responsive **Streamlit Web Application** showcasing cluster exploration, live recommendations, and GenAI query synthesis, packaged for cloud hosting on **Microsoft Azure**.

### 1.2 Business Problem Statement
Streaming users experience *decision fatigue* when browsing large catalogs without clear guidance; research shows that if a user does not find something compelling within 90 seconds, they abandon the session, increasing churn risk. Traditional category tags (e.g., "Dramas" or "Comedies") are overly broad and fail to capture nuances such as dark tone, rapid pacing, cultural context, or director aesthetics.

**Core Objectives:**
- **Thematic Clustering:** Discover latent groupings that go beyond generic tags (e.g., separating gritty true-crime documentaries from international indie dramas).
- **Personalized Discovery:** Provide instant content-based recommendations when a user selects any title.
- **Explainable AI (XAI):** Use Generative AI to generate qualitative reasoning for recommendations.
- **Enterprise Scalability:** Demonstrate Azure Machine Learning Studio integration patterns for continuous pipeline execution and model registry.

---

## 2. Dataset Understanding & Exploratory Data Analysis (EDA)

### 2.1 Dataset Overview
The dataset contains **7,787 titles** and **12 feature columns** representing content added to Netflix up to early 2021:
- `show_id`: Unique identifier for each title.
- `type`: Content category (`Movie` or `TV Show`).
- `title`: Title of the film or show.
- `director`: Director(s) of the content (contains missing values).
- `cast`: Actors and ensemble cast (contains missing values).
- `country`: Production countries (contains missing values).
- `date_added`: Date the title was ingested onto Netflix.
- `release_year`: Year of original release.
- `rating`: Age rating certification (e.g., `TV-MA`, `PG-13`, `TV-14`).
- `duration`: Runtime in minutes (movies) or number of seasons (TV shows).
- `listed_in`: Comma-separated genre classifications.
- `description`: Narrative summary / synopsis.

### 2.2 Key Visualizations and Analytical Insights

#### Figure 1: Distribution of Content Type on Netflix
- **Finding:** Movies make up **69.1% (5,377 titles)** of the catalog, while TV Shows comprise **30.9% (2,410 titles)**.
- **Business Insight:** While movies dominate title count, TV shows generate significantly higher recurring weekly streaming hours. Netflix balances movie volume for immediate gratification with serialized TV shows for long-term subscriber retention.

#### Figure 2: Top 10 Content Producing Countries
- **Finding:** The United States leads with over 2,500 titles, followed by India (~920 titles), the United Kingdom (~400 titles), Canada, France, Japan, and South Korea.
- **Business Insight:** Heavy investment in India, Japan (anime), and South Korea (K-dramas) reflects Netflix's strategic international subscriber growth engine outside saturated North American markets.

#### Figure 3: Growth of Netflix Catalog Over Time (2010 - 2021)
- **Finding:** Exponential ramp-up between 2015 and 2019, peaking with over 2,000 additions annually, before leveling off in 2020 due to global pandemic production delays.
- **Business Insight:** Rapid catalog expansion created information overload, proving the vital need for automated machine-learning-driven discovery systems.

#### Figure 4: Age Rating Distribution Across Content Types
- **Finding:** `TV-MA` (Mature Audiences) is the largest rating category (~37%), followed by `TV-14` (~27%), with family/PG content forming the remainder.
- **Business Insight:** Netflix caters predominantly to adults and young adults, indicating clustering algorithms must segment mature themes, dark psychological narratives, and complex storytelling distinct from child-friendly media.

#### Figure 5: Top 15 Most Common Genres and Category Tags
- **Finding:** `International Movies`, `Dramas`, `Comedies`, `International TV Shows`, and `Documentaries` are the top 5 most frequent tags.
- **Business Insight:** Cross-genre categorization is common (e.g., an Indian film often tagged as "Dramas, International Movies, Independent Movies"), necessitating multi-word n-gram text modeling.

#### Figure 6: Movie Runtime Distribution
- **Finding:** Movie runtimes follow a normal distribution centered at **99 minutes**, with standard deviation of ~28 minutes. Very few movies exceed 160 minutes.
- **Business Insight:** 90–105 minutes represents the optimal consumer viewing sweet spot for streaming features.

---

## 3. Data Cleaning & Feature Engineering

### 3.1 Handling Missing Values
1. **Director (30.7% missing):** Imputed with `'Unknown Director'`.
2. **Cast (9.2% missing):** Imputed with `'Unknown Cast'`.
3. **Country (6.5% missing):** Imputed with `'Unknown Country'`.
4. **Rating (7 missing):** Imputed with the catalog mode (`'TV-MA'`).
5. **Date Added (10 missing):** Imputed using median ingestion date.

### 3.2 Metadata Soup & Text Preprocessing
To build rich content embeddings, we constructed a composite **Metadata Soup** combining:
$$\text{Corpus} = 2 \times \text{Title} + 3 \times \text{Genres} + \text{Director} + \text{Cast} + \text{Country} + \text{Rating} + \text{Description}$$
- **Weighting Rationale:** Repeating `title` (2x) and `genres` (3x) provides higher mathematical weight to explicit categorization and exact title identity during vectorization.
- **NLP Text Pipeline:**
  - Case folding (lowercasing).
  - Punctuation and regex symbol removal.
  - Tokenization and stopword removal (NLTK English stopwords).
  - Word lemmatization using NLTK WordNet Lemmatizer.

### 3.3 TF-IDF Vectorization
- Configured with `max_features=5000`, `ngram_range=(1, 2)` (capturing both single words like "thriller" and bigrams like "true crime", "romantic comedy"), and English stopword elimination.
- Resulting TF-IDF matrix: **7,787 samples $\times$ 5,000 sparse textual features**.

### 3.4 Dimensionality Reduction (PCA)
- High-dimensional sparse TF-IDF spaces suffer from the curse of dimensionality, where distance metrics lose discriminative sensitivity.
- We applied **Principal Component Analysis (PCA)** to project the 5,000 TF-IDF features into the top **50 principal components**, capturing the primary variance axes while eliminating noisy orthographies.
- For visualization, a secondary 2-component PCA projection was fitted for 2D Cartesian scatter plots.

---

## 4. Model Building & Clustering Algorithms

### 4.1 Optimal Cluster Selection ($k$)
We evaluated clustering partitions across $k \in [2, 10]$ using:
1. **Within-Cluster Sum of Squares (WCSS / Elbow Method):** Identified an elbow inflection point between $k=5$ and $k=7$.
2. **Silhouette Coefficient Analysis:** Evaluated cluster compactness versus separation:
   - $k=6$ achieved an optimal trade-off of silhouette score (**0.1245**) and intuitive domain interpretability.

### 4.2 Algorithms Compared
1. **K-Means Clustering ($k=6$):**
   - Convergence: $k$-means++ initialization, 15 iterations.
   - Silhouette Score: **0.1245**
   - Calinski-Harabasz Score: **142.8**
   - Davies-Bouldin Index: **2.18**
2. **Agglomerative Hierarchical Clustering:**
   - Linkage: Ward's minimum variance; Metric: Euclidean.
   - Silhouette Score: **0.1182**
   - Validated that hierarchical trees generate clusters consistent with K-Means centroids.

---

## 5. Interpreted Cluster Themes & Business Personas

| Cluster ID | Assigned Theme Name | % of Catalog | Dominant Genres & Keywords | Target Viewer Persona |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Global International Cinema & Dramas** | ~24.5% | International Movies, Dramas, Foreign Language | Cultural cinema enthusiasts, global cinephiles |
| **1** | **Action, Adventure & Sci-Fi Thrillers** | ~18.2% | Action & Adventure, Sci-Fi, Suspense, Superheroes | Adrenaline seekers, blockbuster fans |
| **2** | **Comedy, Stand-Up & Family Features** | ~21.0% | Stand-Up Comedy, Family Movies, Comedies | Weekend viewers, casual and lighthearted entertainment |
| **3** | **Crime, Mystery & Investigative Docuseries** | ~14.1% | Documentaries, True Crime, Investigative Series | High-engagement binge-watchers, mystery fans |
| **4** | **Romantic Comedies & Indie Dramas** | ~12.4% | Romantic Movies, Independent Cinema, Contemporary Romance | Relationship and romance drama followers |
| **5** | **Kids, Animation & Japanese Anime** | ~9.8% | Children & Family TV, Anime Features, Fantasy | Families with kids, anime and animation fans |

---

## 6. Content-Based Recommendation System

### 6.1 Mathematical Formulation
Using the PCA-transformed feature representations $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{50}$, the semantic similarity between two titles is computed via **Cosine Similarity**:
$$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### 6.2 Recommendation Workflow
1. The user inputs or selects any title from the 7,787 catalog.
2. The system retrieves its 50-dimensional latent embedding vector.
3. Rapid vector dot-product ranking across all catalog vectors retrieves the top $N$ titles with the highest similarity score, excluding the query title itself.
4. Output presents Title, Release Year, Age Rating, Matching Genres, Cluster Theme, and Match Percentage.

---

## 7. GenAI & Microsoft Azure Integration

### 7.1 Microsoft Azure Machine Learning Studio Pipeline
The end-to-end architecture is designed for enterprise deployment on Microsoft Azure:
1. **Azure Data Asset:** `netflix_titles.csv` registered in Azure ML Datastore with version control.
2. **Azure ML Pipeline / Compute:**
   - Automated data preprocessing and TF-IDF extraction running on an Azure ML `Standard_DS3_v2` compute cluster.
   - Model tracking logged using `mlflow` to record WCSS, Silhouette Scores, and cluster distribution artifacts.
3. **Model Registry:** Pickled K-Means, PCA, and TF-IDF models registered with lineage tracking.
4. **Azure Web App / ACI Deployment:** Streamlit application deployed via Docker container on Azure App Service or Azure Container Instances (ACI).

### 7.2 Generative AI Copilot (Google Gemini / Azure OpenAI)
To transform raw similarity numbers into engaging user experiences, we integrated a GenAI copilot:
- **Intelligent Reasoning:** Generates 2-3 sentence explanations explaining why a recommended title shares narrative DNA with the selected movie.
- **Natural Language Thematic Search:** Converts natural user prompts (e.g., *"Show me mind-bending thrillers set in Tokyo"*) into semantic keywords for catalog matching.
- **Resilient Fallback Mode:** Implements an intelligent offline template synthesizer that functions seamlessly even in offline or API-quota-restricted environments.

---

## 8. Streamlit Web Application Features

The production web application features a **curated dark Netflix design system**:
- **Palette:** Cinematic Dark (`#141414`), Netflix Crimson Accent (`#E50914`), Card Charcoal (`#1F1F1F`).
- **Tab 1 - Catalog & Cluster Explorer:** Search titles, filter by cluster or rating, examine cluster composition.
- **Tab 2 - Smart Recommendation Engine:** Select any movie or show, view top 5 recommendations with similarity percentage badges.
- **Tab 3 - GenAI Streaming Assistant:** Ask questions about titles or obtain custom AI explanations.
- **Tab 4 - Analytics & Azure Architecture:** View interactive cluster visualizations, WCSS/Silhouette plots, and deployment schematics.

---

## 9. Final Submission Checklist

- [x] **Project Selected & Dataset Ingested:** Netflix Movies & TV Shows dataset (7,787 records).
- [x] **Exploratory Data Analysis (EDA):** Completed with 8 high-resolution visualizations.
- [x] **Data Preprocessing & NLP Pipeline:** Imputation, text cleaning, lemmatization, TF-IDF vectorization.
- [x] **Dimensionality Reduction:** 50-component PCA variance retention + 2D projection.
- [x] **Unsupervised Model Building:** K-Means ($k=6$), Elbow & Silhouette validation, Hierarchical Clustering comparison.
- [x] **Content-Based Recommender:** Fast cosine similarity matrix over dense embeddings.
- [x] **GenAI & Azure ML Integration:** Explainable AI module and Azure deployment blueprints.
- [x] **Interactive Streamlit Web App:** Fully functional with modern Netflix UI.
- [x] **Formal Documentation:** Word (`.docx`) and Markdown (`.md`) reports prepared.
- [x] **Video Presentation Script & Slides:** 15-20 min English presentation script with camera ON guidelines.
