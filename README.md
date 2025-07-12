# Recommendation System Project No.2
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/reccomd_project2?style=flat-square)
![Jupyter Notebook](https://img.shields.io/badge/jupyter%20notebook-97.6%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/reccomd_project2?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Recommendation System Project No.2* , which provides; 
# 📌 Overview

This project focuses on building a **Recommendation System** for [movie](https://drive.google.com/drive/folders/1cM305e_wLGqGuKnQE2tg0bti8LqekquW?usp=sharing) using a user rating dataset spanning various movie items and users. It leverages collaborative filtering and embedding-based semantic search techniques to generate personalized recommendations.

## 🧩 Problem Statement  

With an overwhelming number of movies available, users often struggle to discover movies aligned with their preferences. If the recommendations do not match users' personal tastes, it can lead to customer dissatisfaction and churn. This project aims to improve user experience by accurately recommending relevant movies based on user behavior and semantic queries, thereby increasing customer satisfaction and reducing churn.

## 🔍 Approach  
The recommendation system employs multiple collaborative filtering (CF) techniques, including user-based and item-based CF, as well as K-Nearest Neighbors (KNN) approaches. Additionally, it integrates semantic search capabilities by embedding user queries using the `all-MiniLM-L6-v2` model to enhance recommendations with context-aware search.

## 🎢 Processes  

1. **Data Loading & Cleaning** – Import user ratings and movie metadata  
2. **Exploratory Data Analysis** – Understand data shape, sparsity, and distribution  
3. **Recommendation Algorithms Implementation** –  
   - User-based CF  
   - Item-based CF  
   - User KNN  
   - Item KNN  
4. **Matrix Factorization** – Latent factor modeling for recommendations  
5. **Semantic Query Integration** – Use sentence embeddings to interpret and match user queries  
6. **Model Saving & Deployment** – Save models for reuse and integrate with a demo app  
7. **Query + Flask API App** – Serve recommendations through API endpoint:  `http://127.0.0.1:5000/recommend?query=comedy`  
8. **Evaluation** – Measure recommendation quality using precision@k metrics  

## 🎯 Results & Impact  

- **Precision@5 (User KNN):** ~30.5%  
- **Precision@5 (Matrix Factorization):** ~28.2%  
- Semantic query integration enables contextually relevant recommendations, improving user satisfaction and search flexibility.

This system aids users in discovering movies tailored to their tastes, reducing choice overload and improving engagement on movie platforms, while also increasing customer satisfaction during system use and boosting opportunities for customers to make purchases.

> Recommendation demo output: [Demo output](Example_result)

> Recommendation with Flask api demo: [Demo flask output](reccomd2_results_with_embedded_query.ipynb)

## ⚙️ Model Development Challenges  

- **Data Sparsity:** Handling the large sparse rating matrix and ensuring meaningful similarity calculations.  
- **Convergence of Factorization Models:** NMF sometimes reached max iteration without convergence, requiring tuning.  
- **Balancing Collaborative Filtering & Semantic Search:** Integrating embeddings to interpret user queries while preserving CF strengths.  
- **Evaluation Metric Sensitivity:** Ensuring precision metrics reflect real user satisfaction and not just algorithmic output.  
- **Serving via API:** Designing a lightweight and accessible Flask API makes it easier for front-end systems or third-party services to interact with the recommendation engine in real-time.  

## 📦 Library Usage
  - `pandas`, `numpy`, `scipy`, `scikit-learn`, `sentence-transformers`, `faiss`, `flask`


## 📝 Example Query  

- Matching with `item_id`
    
      Recommendations for user 435 with query 'Drama Thriller':  
      [3567, 5244, 180497, 7767, 26840, 46723, 33138, 78836, 143365, 93320]

- Matching with `movie_name`
  
      Recommendations for user 319 with query 'Comedy':
      ['Denise Calls Up (1995)', 'Man with the Golden Arm, The (1955)', 'Tag (2018)', "Love Me If You Dare (Jeux d'enfants) (2003)", 'Mr. Skeffington (1944)', 'Top Five (2014)', '12 Angry Men (1997)', 'Very Potter Sequel, A (2010)', 'Grown Ups (2010)', 'Lamerica (1994)']

- Query + Flask api app:  `http://127.0.0.1:5000/recommend?query=comedy`
  
  <img width="1251" height="489" alt="Untitled" src="https://github.com/user-attachments/assets/c2ee0926-e82b-4d35-9b5b-e5f82672b0a4" />
  <img width="1491" height="551" alt="Untitled-1" src="https://github.com/user-attachments/assets/b45f325e-29c2-412f-8c69-bb26318e77e7" />
  
---

