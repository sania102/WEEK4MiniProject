#  Career Path Clustering from Resume Text (Streamlit App)

This is my project  of **Week 4 – Unsupervised Learning** from the AI-ML internship assignment (GNCIPL). It focuses on clustering resume text data into meaningful career paths like **Software**, **Marketing**, **HR**, and **Design**, using KMeans and t-SNE.

# deployment on streamlit link:
https://week4miniprojectsania.streamlit.app/

# google collab link :
https://colab.research.google.com/drive/13Dc03ONoEFJ4JYrQoDsEp9_JXoZrrwEk?usp=sharing


##  Project Overview

| Field                      | Details |
|---------------------------|---------|
| **Project Title**         | Career Path Clustering from Resume Text |
| **Domain**                | HRTech / Recruitment |
| **Data Type**             | Text (Resume summaries) |
| **ML Techniques**         | KMeans Clustering (Unsupervised Learning) |
| **Preprocessing**         | Stopword removal, text cleaning, TF-IDF vectorization |
| **Dimensionality Reduction** | t-SNE for visualization |
| **Clustering Method**     | KMeans |
| **Evaluation Metrics**    | Silhouette Score |
| **Final Output**          | Labeled clusters for resume categories |
| **Tools & Libraries**     | Streamlit, sklearn, nltk, pandas, seaborn, matplotlib |

---

##  Features

-  Paste a resume and get its predicted career cluster.
-  Visualize resume clusters using **t-SNE**.
-  View **Silhouette Score vs. k** to validate cluster count.
-  Clean interface via **Streamlit**.


---

##  How to Run Locally

 pip install -r requirements.txt
 Run the app:
 streamlit run app.py
    

---

##  Use Case

This system helps recruiters and HR departments automatically cluster resume data into role-specific categories, improving candidate screening, job matching, and resume triaging.

---

##  Author

**Sania Shamsi** — AI/ML Intern  
> This project was completed as part of the Week 4 AI/ML internship task using the Unsupervised Learning Playbook provided by GNCIPL.


