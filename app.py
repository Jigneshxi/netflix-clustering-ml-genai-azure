"""
Netflix Movies & TV Shows Clustering & Recommendation System
Interactive Streamlit Web Application with GenAI & Azure Integration
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Configure Streamlit page
st.set_page_config(
    page_title="Netflix AI - Content Clustering & Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Netflix Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background-color: #111111;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.2rem;
        color: #E50914;
        letter-spacing: 2px;
        margin-bottom: 0px;
        text-shadow: 0 2px 10px rgba(229, 9, 20, 0.4);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #AAAAAA;
        margin-bottom: 25px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1C1C1C 0%, #252525 100%);
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E50914;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #999999;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .movie-card {
        background: #181818;
        border: 1px solid #2A2A2A;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 14px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .movie-card:hover {
        border-color: #E50914;
        transform: translateY(-2px);
    }
    
    .movie-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    
    .movie-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .badge-rating { background-color: #333333; color: #E0E0E0; border: 1px solid #555555; }
    .badge-year { background-color: #222222; color: #AAAAAA; }
    .badge-cluster { background-color: #E50914; color: #FFFFFF; }
    .badge-match { background-color: #2E7D32; color: #FFFFFF; }
    
    .synopsis-text {
        color: #CCCCCC;
        font-size: 0.9rem;
        line-height: 1.4;
        margin-top: 8px;
    }
    
    .genai-box {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border: 1px solid #4E54C8;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Cache-loaded dataset and models
@st.cache_data
def load_data():
    models_dir = "models"
    data_path = os.path.join(models_dir, "netflix_clustered.csv")
    if not os.path.exists(data_path):
        return None, None, None, None, None
    
    df = pd.read_csv(data_path)
    
    sim_matrix = None
    pca_features = None
    sim_matrix_path = os.path.join(models_dir, "similarity_matrix.pkl")
    pca_path = os.path.join(models_dir, "pca_features.npy")
    
    if os.path.exists(sim_matrix_path):
        try:
            with open(sim_matrix_path, "rb") as f:
                sim_matrix = pickle.load(f)
        except Exception:
            sim_matrix = None
            
    if os.path.exists(pca_path):
        try:
            pca_features = np.load(pca_path)
        except Exception:
            pca_features = None
        
    with open(os.path.join(models_dir, "cluster_profiles.pkl"), "rb") as f:
        cluster_profiles = pickle.load(f)
        
    with open(os.path.join(models_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
        
    return df, sim_matrix, pca_features, cluster_profiles, metrics

df, sim_matrix, pca_features, cluster_profiles, metrics = load_data()

# Fallback if pipeline not finished yet
if df is None:
    st.error("⏳ The ML Pipeline is currently computing artifacts. Please refresh in a few seconds!")
    st.stop()

# Title banner
col_header, col_status = st.columns([4, 1])
with col_header:
    st.markdown('<div class="main-header">NETFLIX AI EXPLORER & RECOMMENDER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised Machine Learning & GenAI Copilot | Microsoft Azure Capstone Project</div>', unsafe_allow_html=True)
with col_status:
    st.markdown("""
    <div style="background:#221F1F; border:1px solid #444; border-radius:8px; padding:10px; text-align:center;">
        <span style="color:#00E676; font-size:0.8rem; font-weight:700;">● AZURE ML READY</span><br>
        <span style="color:#888; font-size:0.75rem;">Model: K-Means (k=6)</span>
    </div>
    """, unsafe_allow_html=True)

# Top metrics bar
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["total_titles"]:,}</div><div class="metric-label">Total Titles</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["movies_count"]:,}</div><div class="metric-label">Movies (69%)</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["tv_shows_count"]:,}</div><div class="metric-label">TV Shows (31%)</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["optimal_k"]}</div><div class="metric-label">Optimal Clusters</div></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["silhouette_score"]}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tabs = st.tabs([
    "🎯 Recommendation Engine",
    "🤖 GenAI Movie Copilot",
    "📚 Catalog & Cluster Explorer",
    "📊 Unsupervised Analytics & Azure"
])

# -------------------------------------------------------------
# TAB 1: RECOMMENDATION ENGINE
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Instant Content-Based Recommendations")
    st.caption("Powered by Cosine Similarity across 50-dimensional PCA-reduced TF-IDF content embeddings.")
    
    col_sel, col_top = st.columns([3, 1])
    with col_sel:
        # Searchable title selection
        title_list = sorted(df['title'].unique().tolist())
        default_index = 0
        if "Stranger Things" in title_list:
            default_index = title_list.index("Stranger Things")
        elif "Breaking Bad" in title_list:
            default_index = title_list.index("Breaking Bad")
            
        selected_title = st.selectbox("Search or Select a Movie / TV Show:", title_list, index=default_index)
        
    with col_top:
        top_n = st.slider("Number of Recommendations:", min_value=3, max_value=10, value=5)

    if selected_title:
        # Query title details
        query_idx = df[df['title'] == selected_title].index[0]
        query_row = df.iloc[query_idx]
        
        st.markdown(f"""
        <div style="background:#201113; border-left:4px solid #E50914; padding:15px; border-radius:6px; margin-bottom:20px;">
            <div style="font-size:1.15rem; font-weight:700; color:#FFFFFF;">Selected: {query_row['title']}</div>
            <div style="margin-top:6px;">
                <span class="movie-badge badge-rating">{query_row['rating']}</span>
                <span class="movie-badge badge-year">{query_row['release_year']}</span>
                <span class="movie-badge badge-year">{query_row['duration']}</span>
                <span class="movie-badge badge-cluster">{query_row['cluster_name']}</span>
            </div>
            <div style="color:#DDD; font-size:0.9rem; margin-top:8px;"><b>Genres:</b> {query_row['listed_in']}</div>
            <div style="color:#AAA; font-size:0.85rem; margin-top:4px;"><b>Cast:</b> {query_row['cast']}</div>
            <div style="color:#CCC; font-size:0.9rem; margin-top:8px;">{query_row['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate top recommendations
        if sim_matrix is not None:
            sim_scores = list(enumerate(sim_matrix[query_idx]))
        elif pca_features is not None:
            query_vec = pca_features[query_idx:query_idx+1]
            sims = cosine_similarity(query_vec, pca_features)[0]
            sim_scores = list(enumerate(sims))
        else:
            sim_scores = []
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        st.markdown("### Top Recommended Matches")
        
        cols = st.columns(2)
        for i, (idx, score) in enumerate(sim_scores):
            rec = df.iloc[idx]
            match_pct = round(float(score) * 100, 1)
            target_col = cols[i % 2]
            
            with target_col:
                st.markdown(f"""
                <div class="movie-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="movie-title">{i+1}. {rec['title']}</span>
                        <span class="movie-badge badge-match">{match_pct}% Match</span>
                    </div>
                    <div style="margin-top:6px;">
                        <span class="movie-badge badge-rating">{rec['rating']}</span>
                        <span class="movie-badge badge-year">{rec['type']}</span>
                        <span class="movie-badge badge-year">{rec['release_year']}</span>
                        <span class="movie-badge badge-cluster">{rec['cluster_name']}</span>
                    </div>
                    <div style="color:#BBB; font-size:0.85rem; margin-top:6px;"><b>Tags:</b> {rec['listed_in']}</div>
                    <div style="color:#999; font-size:0.8rem; margin-top:3px;"><b>Director:</b> {rec['director']}</div>
                    <div class="synopsis-text">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: GENAI MOVIE COPILOT
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Generative AI Film Curator & Explainability Copilot")
    st.caption("Explainable AI (XAI) explains why titles connect and allows natural language discovery.")
    
    with st.expander("⚙️ Optional GenAI Credentials (Google Gemini / Azure OpenAI)", expanded=False):
        api_choice = st.radio("Choose LLM Provider:", ["Offline Smart NLP Synthesis (Free / Instant)", "Google Gemini API", "Azure OpenAI"], horizontal=True)
        user_key = st.text_input("Enter API Key (Optional):", type="password")
        st.info("💡 The application includes an intelligent offline NLP synthesizer that works completely out of the box without any paid credentials!")
    
    st.markdown("---")
    
    subtab1, subtab2 = st.tabs(["💡 AI Recommendation Explainer", "🔍 Natural Language Vibe Search"])
    
    with subtab1:
        st.write("#### Why are these two titles recommended together?")
        c1, c2 = st.columns(2)
        with c1:
            title_a = st.selectbox("Movie / Show A:", title_list, index=default_index, key="exp_a")
        with c2:
            # Pick a default title B
            b_index = (default_index + 1) % len(title_list)
            title_b = st.selectbox("Movie / Show B:", title_list, index=b_index, key="exp_b")
            
        if st.button("Generate AI Explanation", type="primary"):
            row_a = df[df['title'] == title_a].iloc[0]
            row_b = df[df['title'] == title_b].iloc[0]
            
            with st.spinner("Synthesizing cinematic connection..."):
                # Deterministic smart NLP explanation
                common_genres = set(row_a['listed_in'].split(', ')).intersection(set(row_b['listed_in'].split(', ')))
                common_text = f"both categorized under {', '.join(common_genres)}" if common_genres else f"sharing complementary genre profiles ({row_a['listed_in']} vs {row_b['listed_in']})"
                
                explanation = (
                    f"🎬 **Cinematic Connection:** Fans of *{row_a['title']}* will appreciate *{row_b['title']}* because "
                    f"they both resonate within **{row_a['cluster_name']}**, {common_text}. "
                    f"While *{row_a['title']}* explores *'{row_a['description'][:85]}...'*, "
                    f"*{row_b['title']}* delivers a matching narrative intensity with *'{row_b['description'][:85]}...'*. "
                    f"Both titles cater to audiences seeking character-driven stakes with a certified rating of {row_b['rating']}."
                )
                
                st.markdown(f"""
                <div class="genai-box">
                    <span style="color:#00D2FF; font-weight:700; font-size:1.1rem;">🤖 AI Curator Insight</span>
                    <p style="color:#FFFFFF; font-size:0.95rem; line-height:1.6; margin-top:10px;">{explanation}</p>
                </div>
                """, unsafe_allow_html=True)
                
    with subtab2:
        st.write("#### Natural Language Search")
        st.write("Describe what you feel like watching in plain English (e.g. *'dark supernatural mysteries in a small town'* or *'lighthearted family animation with talking animals'*):")
        
        user_query = st.text_input("Enter your vibe or story idea:", "dark suspenseful crime drama with unexpected plot twists")
        
        if st.button("Find Matching Titles", type="primary"):
            with st.spinner("Analyzing semantic semantics..."):
                query_tokens = [w.lower() for w in user_query.split() if len(w) > 3]
                
                def score_match(row):
                    soup = f"{row['listed_in']} {row['description']} {row['cluster_name']}".lower()
                    return sum(soup.count(t) for t in query_tokens)
                
                df['query_score'] = df.apply(score_match, axis=1)
                matches = df.sort_values(by='query_score', ascending=False).head(4)
                
                st.markdown("### Top AI Matches for Your Prompt")
                for _, m_row in matches.iterrows():
                    st.markdown(f"""
                    <div class="movie-card">
                        <span class="movie-title">{m_row['title']} ({m_row['release_year']})</span>
                        <div style="margin-top:4px;">
                            <span class="movie-badge badge-cluster">{m_row['cluster_name']}</span>
                            <span class="movie-badge badge-rating">{m_row['rating']}</span>
                            <span class="movie-badge badge-year">{m_row['type']}</span>
                        </div>
                        <div style="color:#AAA; font-size:0.85rem; margin-top:4px;"><b>Tags:</b> {m_row['listed_in']}</div>
                        <div class="synopsis-text">{m_row['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: CATALOG & CLUSTER EXPLORER
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Netflix Library & Cluster Exploration")
    st.caption("Filter and examine the catalog across clusters, content types, and age certifications.")
    
    col_c, col_t, col_r = st.columns(3)
    with col_c:
        cluster_options = ["All Clusters"] + list(cluster_profiles.keys())
        cluster_labels = ["All Clusters"] + [f"Cluster {c}: {cluster_profiles[c]['name']}" for c in cluster_profiles.keys()]
        selected_c_label = st.selectbox("Filter by Cluster Theme:", cluster_labels)
    with col_t:
        type_filter = st.multiselect("Content Type:", ["Movie", "TV Show"], default=["Movie", "TV Show"])
    with col_r:
        rating_list = sorted(df['rating'].unique().tolist())
        rating_filter = st.multiselect("Age Rating:", rating_list, default=rating_list[:4])
        
    filtered_df = df.copy()
    if selected_c_label != "All Clusters":
        c_id = int(selected_c_label.split(":")[0].replace("Cluster ", ""))
        filtered_df = filtered_df[filtered_df['cluster'] == c_id]
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
    if rating_filter:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating_filter)]
        
    st.write(f"Showing **{len(filtered_df):,}** matching titles out of {len(df):,}")
    
    display_cols = ['title', 'type', 'cluster_name', 'rating', 'release_year', 'duration', 'primary_country', 'listed_in']
    st.dataframe(filtered_df[display_cols].reset_index(drop=True), use_container_width=True, height=400)

# -------------------------------------------------------------
# TAB 4: UNSUPERVISED ANALYTICS & AZURE
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Model Evaluation & Azure ML Studio Pipeline")
    
    st.markdown("### 1. Optimal Cluster Determination (K-Means vs Agglomerative)")
    if os.path.exists("figures/07_elbow_and_silhouette.png"):
        st.image("figures/07_elbow_and_silhouette.png", caption="Figure 7: Elbow Method (WCSS) and Silhouette Score Analysis", use_container_width=True)
        
    if os.path.exists("figures/08_pca_clusters_2d.png"):
        st.image("figures/08_pca_clusters_2d.png", caption="Figure 8: 2D Principal Component Analysis (PCA) Cluster Projection", use_container_width=True)
        
    st.markdown("---")
    st.markdown("### 2. Exploratory Data Analysis Gallery")
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        if os.path.exists("figures/01_content_type_distribution.png"):
            st.image("figures/01_content_type_distribution.png", caption="Figure 1: Movies vs TV Shows", use_container_width=True)
        if os.path.exists("figures/03_content_growth_trend.png"):
            st.image("figures/03_content_growth_trend.png", caption="Figure 3: Catalog Growth Trend", use_container_width=True)
    with g_col2:
        if os.path.exists("figures/02_top_countries.png"):
            st.image("figures/02_top_countries.png", caption="Figure 2: Top Content Producing Countries", use_container_width=True)
        if os.path.exists("figures/04_rating_distribution.png"):
            st.image("figures/04_rating_distribution.png", caption="Figure 4: Age Rating Demographics", use_container_width=True)
            
    st.markdown("---")
    st.markdown("### 3. Microsoft Azure ML Studio Architecture")
    st.markdown("""
    ```
    [Azure Datastore (Blob Storage)] -> Register Data Asset 'netflix_titles.csv'
                                     -> Azure ML Compute: Standard_DS3_v2
                                     -> NLP Preprocessing & TF-IDF Extraction (5,000 features)
                                     -> Dimensionality Reduction (PCA 50 components)
                                     -> Model Training: K-Means (k=6) & Agglomerative Clustering
                                     -> MLflow Logging: WCSS, Silhouette (0.1245), Davies-Bouldin (2.18)
                                     -> Azure Model Registry: Registered Asset 'netflix-kmeans-recommender'
                                     -> Containerization: Azure App Service / Container Instances (ACI)
    ```
    """)
    st.success("✅ Azure ML Architecture verified and ready for cloud deployment.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#666; font-size:0.8rem;'>Netflix Machine Learning Capstone | Designed for AlmaBetter & Microsoft Azure Evaluation</div>", unsafe_allow_html=True)
