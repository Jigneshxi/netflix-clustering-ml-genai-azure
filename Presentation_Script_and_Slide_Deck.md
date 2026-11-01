# AlmaBetter Capstone Video Presentation Script & Interview Preparation
## Netflix Movies & TV Shows Clustering and Recommendation System
**Program:** Master's in Data Science & AI with Microsoft Azure  
**Video Length Constraint:** Strictly between **15 minutes** and **25 minutes** (rejection if < 15 mins)  
**Language:** English (Fluent & Grammatically Accurate — 5 marks rubric)  
**Camera:** Mandatory Camera ON (Face clearly visible in Picture-in-Picture or split screen)  

---

## 🧭 Pre-Recording Checklist
- [ ] Good front-facing lighting with webcam at eye level.
- [ ] Quiet room with crisp microphone input.
- [ ] Browser tabs open and ready:
  1. Slide Deck / Notion presentation page.
  2. Google Colab / Jupyter Notebook (`Netflix_Movies_and_TV_Shows_Clustering_AlmaBetter.ipynb`) with pre-rendered plots.
  3. Streamlit Web App (`http://localhost:8501`) live demo.
  4. Microsoft Azure ML Studio architecture diagram.
- [ ] Timer running on your mobile/screen to hit between 16:00 and 19:00 minutes.

---

## ⏱️ Video Breakdown (Aligned with Official AlmaBetter Guidelines)

| Section # | AlmaBetter Guideline Section | Target Time | Cumulative |
| :---: | :--- | :---: | :---: |
| **1** | **Introduction** | 1.5 min | 00:00 – 01:30 |
| **2** | **Problem Understanding** | 2.0 min | 01:30 – 03:30 |
| **3** | **Data Understanding & Exploratory Analysis** | 2.5 min | 03:30 – 06:00 |
| **4** | **Data Preprocessing & Feature Preparation** | 2.5 min | 06:00 – 08:30 |
| **5** | **Model Building & Evaluation (2+ Algorithms)** | 2.5 min | 08:30 – 11:00 |
| **6** | **Azure / GenAI / Streamlit Deployment Integration** | 2.5 min | 11:00 – 13:30 |
| **7** | **Challenges, Optimization & Improvements** | 1.5 min | 13:30 – 15:00 |
| **8** | **Learnings & Business Impact** | 1.5 min | 15:00 – 16:30 |
| **Bonus** | **Comprehensive Answers to 5 Follow-Up Interview Questions** | 2.5 min | 16:30 – 19:00 |

---

## 🎙️ Word-for-Word English Presentation Script

### 1. Introduction (00:00 – 01:30)
**[Camera: Full face or Picture-in-Picture. Confident, smiling tone.]**

> *"Hello respected mentors and evaluators. Welcome to my Capstone Project presentation for the Master's in Data Science and AI program with Microsoft Azure.*
>
> *My name is [Your Name], and today I am excited to present my project:* **Unsupervised Machine Learning: Netflix Movies and TV Shows Clustering and Content Recommendation with Microsoft Azure and Generative AI Integration**.*
>
> *The goal of this project is to analyze Netflix's vast content catalog of nearly 7,800 titles, investigate how Netflix's content strategy has shifted between movies and TV shows since 2010, segment similar content using natural language processing and unsupervised clustering, and deploy an interactive recommendation engine powered by Explainable Generative AI.*
>
> *Our technical stack comprises Python, Pandas, Scikit-Learn, NLTK, Streamlit for modern web app deployment, and Microsoft Azure Machine Learning Studio for cloud pipeline tracking and registry. Over the next 17 minutes, I will guide you step-by-step through our business objectives, exploratory data analysis, hypothesis testing, NLP feature engineering, clustering algorithms, and a live web application demo."*

---

### 2. Problem Understanding (01:30 – 03:30)
**[Slide: Problem Statement & Flixable Report Context]**

> *"Let us examine the business problem. In the streaming entertainment economy, the most critical currency is subscriber attention and retention. When a subscriber opens Netflix, if they spend more than 60 to 90 seconds scrolling through titles without finding something compelling, they experience **decision fatigue** and abandon the session. Repeated decision fatigue leads directly to customer churn.*
>
> *Furthermore, a seminal 2018 industry report by Flixable revealed a fascinating transformation: while Netflix's movie catalog decreased by more than 2,000 titles between 2010 and 2018, its volume of TV shows nearly tripled. This poses an essential strategic question: **Why is Netflix prioritizing TV series over standalone movies, and how can we better organize this expanding library?***
>
> *Traditional categorization relies on broad, rigid genres like 'Drama' or 'Action'. But a dark psychological Nordic thriller and an upbeat family legal drama cannot both be lumped into generic 'Dramas'. By employing Unsupervised Machine Learning, we can discover latent thematic clusters that capture tone, pacing, cultural context, and narrative nuance—empowering Netflix's recommendation engine to combat search fatigue and drive subscriber lifetime value."*

---

### 3. Data Understanding & Exploratory Analysis (03:30 – 06:00)
**[Slide: Key Visualizations & Hypotheses]**

> *"Let us look at the data. The dataset consists of 7,787 rows and 12 feature columns representing titles on Netflix. Here are our core EDA findings:*
>
> *1. **Content Type Composition:** Movies account for 69.1% (5,377 titles) while TV Shows comprise 30.9% (2,410 titles). While movies provide high initial title variety, serialized TV shows drive recurring weekly streaming hours.*
>
> *2. **Geographic Distribution:** The United States is the primary producer with over 2,500 titles, followed by India (~920 titles), the UK, Canada, France, Japan, and South Korea. This demonstrates Netflix's aggressive internationalization strategy.*
>
> *3. **Catalog Addition Trends & Flixable Hypothesis:** Content additions surged exponentially from 2015 to 2019, peaking in late 2019 with over 2,000 additions annually. We formulated a Chi-Square test of independence to statistically test the Flixable report: the test yielded a p-value less than 0.001, confirming a statistically significant acceleration toward TV shows in recent years!*
>
> *4. **Maturity Rating Dynamics:** TV-MA (Mature Audiences) and TV-14 represent over 60% of all content, showing that Netflix's core viewer base is adult and young adult.*
>
> *5. **Runtime Distributions:** Feature films follow a normal Gaussian curve centered at 99.3 minutes (median: 98 minutes), establishing that the 90-to-105-minute window is the commercial sweet spot for consumer attention."*

---

### 4. Data Preprocessing & Feature Preparation (06:00 – 08:30)
**[Slide: Preprocessing Flowchart & NLP Pipeline]**

> *"Now, let's discuss our data preprocessing and feature engineering steps.*
>
> *First, handling missing values: In real-world multimedia datasets, missing metadata is prevalent. Director was missing in ~30.7% of titles, Cast in 9.2%, and Country in 6.5%. Instead of deleting rows—which would destroy nearly 3,000 valid titles—we used informative domain imputation: 'Unknown Director', 'Unknown Cast', and 'Unknown Country'. Missing ratings were imputed with the catalog mode ('TV-MA').*
>
> *Next, we formulated our **Metadata Soup**: concatenating title, genres, director, cast, country, rating, and description into a unified text string, assigning double weight to title and triple weight to genres.*
>
> *Our NLP cleaning pipeline:*
> - *Lowercasing and regex removal of punctuation and special characters.*
> - *Tokenization and filtering against NLTK English stopwords.*
> - *Lemmatization using WordNet Lemmatizer to reduce inflected words to their root lemma.*
>
> *We then transformed the cleaned text using **TF-IDF Vectorization** (5,000 features, unigrams and bigrams). To conquer the *curse of dimensionality*—where Euclidean distance becomes equidistant in high-dimensional sparse spaces—we applied **Principal Component Analysis (PCA)** to project down to the top 50 orthogonal principal components, capturing dense variance and eliminating noise."*

---

### 5. Model Building & Evaluation (08:30 – 11:00)
**[Slide: Elbow Method, Silhouette Plot, and Cluster Evaluation]**

> *"To meet the rubric criteria of implementing multiple algorithms, we evaluated two distinct unsupervised clustering techniques:*
>
> *1. **K-Means Clustering:** We evaluated $k$ from 2 to 10. Using the **Elbow Method (Within-Cluster Sum of Squares)**, we observed a distinct bend at $k=6$. Evaluating the **Silhouette Coefficient**, $k=6$ achieved an optimal score of **0.1562**, with a Calinski-Harabasz score of **499.44** and a Davies-Bouldin index of **2.2162**.*
>
> *2. **Agglomerative Hierarchical Clustering:** We evaluated hierarchical clustering using Ward's minimum variance linkage with Euclidean metric, visualized via a Dendrogram. It produced a Silhouette Score of **0.1314**, validating the cluster boundaries discovered by K-Means.*
>
> *Our 6 resulting clusters represent distinct business personas:*
> - *Cluster 0: Global International Cinema & Dramas (48.7% of catalog)*
> - *Cluster 1: Action, Adventure & Sci-Fi Thrillers (6.4%)*
> - *Cluster 2: Comedy, Stand-Up & Family Entertainment (9.8%)*
> - *Cluster 3: Crime, Mystery & Docuseries (20.2%)*
> - *Cluster 4: Romantic Comedies & Indie Stories (6.8%)*
> - *Cluster 5: Kids, Animation & Japanese Anime (8.1%)*
>
> *Using these embeddings, we also built a Content-Based Recommender using Cosine Similarity, generating top-5 recommendations in under 50 milliseconds."*

---

### 6. Azure / GenAI / Streamlit Deployment Integration (11:00 – 13:30)
**[Screen Share: Live Streamlit Application Demo]**

> *"Let me now demonstrate our live, production-grade Streamlit Web Application.*
>
> *[Show Streamlit App on http://localhost:8501]: As you can see, the interface is styled with Netflix's iconic dark aesthetic—deep obsidian backgrounds, Netflix red accent `#E50914`, and card layouts.*
>
> *In **Tab 1: Recommendation Engine**, a user can select any title—like 'Stranger Things' or 'Dark'—and click 'Find Similar Titles'. The app computes cosine similarity across our 50-dimensional PCA embeddings and returns the top 5 recommendations complete with percentage match badges, age ratings, and synopses.*
>
> *In **Tab 2: GenAI Movie Copilot**, we integrated Explainable AI. Traditional recommender systems give users cold numbers like '95% match'. Our GenAI Copilot uses the Google Gemini API to synthesize natural language explanations of why the titles connect—for example, explaining shared atmospheric dread, supernatural mystery, and 1980s retro aesthetics. It also allows natural language 'vibe searches' like 'dark crime thrillers with shocking twists'.*
>
> *In **Tab 3 & 4**, users can explore the catalog across clusters and view our Microsoft Azure ML Studio architecture diagram—showing automated data ingestion, MLflow experiment tracking, and model registration."*

---

### 7. Challenges, Optimization & Improvements (13:30 – 15:00)
**[Slide: Technical Challenges & Solutions]**

> *"During this project, we overcame three core engineering challenges:*
>
> *1. **High-Dimensional Text Sparsity:** TF-IDF text features caused distance metric saturation in K-Means. We solved this by using 50-component PCA dimensionality reduction, which improved clustering convergence and lifted the silhouette score.*
>
> *2. **Extensive Missing Metadata:** With 30% of directors and 9% of cast missing, naive row removal would have discarded one-third of the dataset. We developed domain-appropriate placeholder imputation that preserved full catalog integrity.*
>
> *3. **Unsupervised Metric Ambiguity:** Since unsupervised learning lacks ground-truth labels, tuning $k$ required multi-metric triangulation—cross-referencing WCSS Elbow, Silhouette Scores, Calinski-Harabasz variance, and domain interpretability.*
>
> *For future improvements, we can combine content features with collaborative filtering user clickstream telemetry, utilize Transformer embeddings like Sentence-BERT, and deploy automated drift monitoring in Azure ML."*

---

### 8. Learnings & Business Impact (15:00 – 16:30)
**[Slide: Strategic ROI & Takeaways]**

> *"To summarize the business value of this Capstone Project:*
> - *We mathematically verified the Flixable hypothesis, showing how Netflix pivoted to serialized TV shows for subscriber retention.*
> - *We transformed unstructured catalog metadata into 6 distinct, actionable content clusters.*
> - *Our sub-second cosine recommendation engine directly mitigates subscriber decision fatigue.*
> - *Catalog gap analysis identified that Anime and Family content accounts for under 10% of titles, offering a clear growth vector for content acquisition.*
> - *And our GenAI Copilot bridges the gap between black-box machine learning and subscriber trust."*

---

### Bonus: Expert Answers to 5 Follow-Up Interview Questions (16:30 – 19:00)
**[Camera: Full face, engaging and articulate delivery]**

> *"To conclude, I would like to address the 5 key technical questions an interviewer might ask about this project:*
>
> ***Question 1: Why did you choose K-Means and Agglomerative Clustering over other algorithms?***  
> *Answer: K-Means with k-means++ initialization provides $O(n \cdot k \cdot d)$ computational efficiency, which scales seamlessly to thousands of catalog titles. To ensure our spherical K-Means centroids did not impose artificial geometric structures, we benchmarked against Agglomerative Hierarchical Clustering using Ward's minimum variance linkage. The high agreement in silhouette scores (0.156 vs 0.131) validated that our 6 clusters reflect genuine semantic groupings.*
>
> ***Question 2: How would you improve the model further?***  
> *Answer: First, by upgrading from static TF-IDF to contextual dense vector embeddings using pre-trained Sentence-BERT or OpenAI text-embedding-3-small. Second, by implementing a hybrid recommendation system combining content-based filtering with collaborative filtering matrix factorization (such as Alternating Least Squares on subscriber watch logs).*
>
> ***Question 3: Which features were most important and why?***  
> *Answer: The `listed_in` (genres) and `description` (synopses) features carried the highest discriminative power, followed by `director` and `cast`. This is why we created a weighted Metadata Soup, applying triple weighting to genres and double weighting to titles to ensure explicit categorization anchors the clustering.*
>
> ***Question 4: How would you handle noisy or imbalanced data?***  
> *Answer: Text noise was sanitized via our NLP pipeline—lowercasing, regex non-alphanumeric cleaning, and lemmatization. For class imbalances (such as movies outnumbering TV shows 2-to-1), TF-IDF naturally balances rare and frequent terms via Inverse Document Frequency, and PCA centers and scales the features orthogonally.*
>
> ***Question 5: How would you deploy this solution for real users at scale?***  
> *Answer: In an enterprise setting, I would containerize the Streamlit app using Docker, store embeddings in a vector database like Azure Cosmos DB or Pinecone, register the models in Azure ML Model Registry, and host the web application on Azure App Service with auto-scaling.*
>
> *Thank you very much for your time, consideration, and feedback!"*
