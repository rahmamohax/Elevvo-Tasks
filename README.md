# Elevvo NLP Internship Tasks

This repository contains my completed tasks from the **Elevvo Internship Program** in the **Natural Language Processing (NLP) track**.  
The tasks focus on text preprocessing, classification, topic modeling, and practical NLP applications using various datasets and machine learning techniques.

---

## 📂 Table of Contents
- [Task 2: News Category Classification](#task-2-news-category-classification)
- [Task 3: Fake News Detection](#task-3-fake-news-detection)
- [Task 5: Topic Modeling on News Articles](#task-5-topic-modeling-on-news-articles)
- [Task 8: Resume Screening Using NLP](#task-8-resume-screening-using-nlp)
- [Tools & Libraries](#tools--libraries)
- [How to Run](#how-to-run)
- [Repository Link](#repository-link)

---

## **Task 2: News Category Classification**
**Dataset:** [AG News Dataset (Kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  
**Goal:** Classify news articles into categories such as sports, business, politics, technology, etc.

**Steps:**
1. Preprocess the text:
   - Tokenization
   - Stopword removal
   - Lemmatization
2. Vectorize text using **TF-IDF** or **word embeddings**.
3. Train a **multiclass classifier**:
   - Logistic Regression
   - Random Forest
   - SVM
4. Visualize the most frequent words per category using:
   - Bar plots
   - Word clouds
5. Bonus: Train a simple feedforward **Neural Network** with Keras.

---

## **Task 3: Fake News Detection**
**Dataset:** [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
**Goal:** Classify news articles as **real** or **fake**.

**Steps:**
1. Preprocess titles and content:
   - Remove stopwords
   - Lemmatize
   - Vectorize (TF-IDF or embeddings)
2. Train classifiers:
   - Logistic Regression
   - SVM
3. Evaluate using:
   - Accuracy
   - F1-score
4. **Bonus:** Use word clouds to visualize common terms in **fake** vs **real** news.

**Tools:** Python, Pandas, NLTK/spaCy, Scikit-learn

---

## **Task 5: Topic Modeling on News Articles**
**Dataset:** [BBC News Dataset (Kaggle)](https://www.kaggle.com/datasets/pariza/bbc-news-summary)  
**Goal:** Discover hidden topics or themes in a collection of news articles or blog posts.

**Steps:**
1. Preprocess the text:
   - Tokenization
   - Lowercasing
   - Stopword removal
2. Apply **Latent Dirichlet Allocation (LDA)** to extract dominant topics.
3. Display the most significant words per topic.

---

## **Task 8: Resume Screening Using NLP**
**Dataset:**  
- Resume Dataset (Kaggle)  
- Job Description Dataset (Kaggle)

**Goal:** Develop a system to **screen and rank resumes** based on job descriptions.

**Steps:**
1. Preprocess resumes and job descriptions using embeddings.
2. Match resumes to job descriptions using:
   - Cosine similarity
   - Classification-based matching
3. Present top-ranked resumes with **brief justifications**.

---

## **Tools & Libraries**
- Python
- Pandas
- NumPy
- NLTK / spaCy
- Scikit-learn
- Matplotlib / Seaborn
- WordCloud
- Keras / TensorFlow

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/rahmamohax/Elevvo-Tasks.git
