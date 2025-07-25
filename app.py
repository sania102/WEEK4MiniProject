import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# 1. Loading the Sample Resume Dataset

data = {
    "Resume": [
        "Software engineer with experience in Python, Java, and backend development.",
        "Marketing specialist in digital campaigns, SEO, and content writing.",
        "Data scientist skilled in Python, machine learning, and data visualization.",
        "HR executive experienced in talent acquisition and employee engagement.",
        "Front-end developer with HTML, CSS, React, and JavaScript expertise.",
        "Financial analyst with Excel, forecasting, and budgeting experience.",
        "UX designer focused on user research, prototyping, and wireframing.",
        "Network engineer managing routers, switches, and network security.",
        "Content writer skilled in blogs, SEO, storytelling, and proofreading.",
        "AI/ML engineer with TensorFlow, PyTorch, and deep learning knowledge."
    ]
}
df = pd.DataFrame(data)


# 2. Clean Text

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

df['Cleaned'] = df['Resume'].apply(clean_text)


# 3. Vectorizing & Cluster

vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['Cleaned']).toarray()

# Optimal k using silhouette
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

cluster_labels = {
    0: 'HR & Finance',
    1: 'Software & AI/ML',
    2: 'Marketing & Content',
    3: 'Frontend & Design'
}
df['Label'] = df['Cluster'].map(cluster_labels)


# 4. Streamlit Interface

st.title("Career Path Clustering App")
st.markdown("Paste a resume to predict its career cluster and visualize the groupings.")

# 📄 Resume Input
user_input = st.text_area("Paste Resume Text Here", height=200)

if st.button("Predict Career Path"):
    if user_input.strip() == "":
        st.warning("Please enter resume text.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = kmeans.predict(vector)[0]
        label = cluster_labels[prediction]
        st.success(f" Predicted Career Path Cluster: **{label}**")

#  Silhouette Score Plot

if st.checkbox(" Show Silhouette Score Plot"):
    silhouette_scores = []
    k_range = range(2, 8)
    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42).fit(X)
        score = silhouette_score(X, km.labels_)
        silhouette_scores.append(score)

    fig2, ax2 = plt.subplots()
    ax2.plot(list(k_range), silhouette_scores, marker='o', color='green')
    ax2.set_title("Silhouette Score vs Number of Clusters")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Silhouette Score")
    st.pyplot(fig2)


#  t-SNE Cluster Visualization

if st.checkbox(" Show t-SNE Resume Clusters"):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(X)
    df['x-tsne'] = tsne_results[:, 0]
    df['y-tsne'] = tsne_results[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='x-tsne', y='y-tsne', hue='Label', palette='Set2', s=100, ax=ax)
    ax.set_title("Resume Clusters (via t-SNE)")
    st.pyplot(fig)

