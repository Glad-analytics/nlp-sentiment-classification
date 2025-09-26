# nlp-sentiment-classification
NLP Sentiment Classification project using IMDB dataset. Includes preprocessing (lemmatization, stopword removal), exploratory analysis, and model training with Logistic Regression, Naive Bayes, MLP, and CNN — evaluated using ROC-AUC and PR-AUC.
This project applies **Natural Language Processing (NLP)** techniques to classify IMDB movie reviews as **positive or negative**. It compares **traditional machine learning models** with **deep learning models** to evaluate performance on sentiment classification.

## 📌 Project Overview
- **Dataset:** IMDB Large Movie Review Dataset (50,000 reviews, evenly split positive/negative).  
- **Traditional ML Models:** Logistic Regression, Multinomial Naïve Bayes.  
- **Deep Learning Models:** Feedforward Neural Network (MLP), Convolutional Neural Network (CNN).  
- **Preprocessing:** Lowercasing, cleaning, tokenisation, lemmatisation, stopword removal, TF-IDF vectorisation, and frequency analysis with word clouds.  

## ⚙️ Tools & Libraries
- Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras, SpaCy, Gensim, Matplotlib, WordCloud)  
- Jupyter Notebook for implementation and analysis  

## 📊 Key Results
- **Logistic Regression:** Accuracy & F1 = 0.88  
- **Naïve Bayes:** Accuracy & F1 = 0.85  
- **MLP:** Accuracy & F1 = 0.86  
- **CNN:** Accuracy = 0.85  

✅ Achieved performance comparable to or exceeding prior benchmarks (Maas et al., 2011 baseline = 0.88).  

## 📂 Repository Contents
- `NLP_Sentiment_Analysis.ipynb` → Jupyter Notebook with full code and visualisations  
- `NLP_Sentiment_Analysis.pdf` → Full project report (academic format)
   
## 🔗 Quick Links
- [View Notebook](./NLP_Sentiment_Analysis.ipynb)  
- [View Report](./NLP_Sentiment_Analysis.pdf)
  
## 🚀 How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/Glad-analytics/nlp-sentiment-classification.git
2.	Open NLP_Sentiment_Analysis.ipynb in Jupyter Notebook.
3.	Install dependencies with: 'pip install -r requirements.txt'
4.	Run the notebook step by step to reproduce preprocessing, model training, and evaluation.
