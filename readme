# 📄 Resume Screening and Classification System

This project demonstrates the complete pipeline for an **AI-based Resume Classification System** using traditional Machine Learning models (Random Forest, Naive Bayes), Deep Learning (LSTM), and Transformer-based models (BERT, DistilBERT). The system classifies resumes into predefined job categories such as Data Science, Web Development, HR, etc.

---

## 🔍 Problem Statement

Manually reviewing resumes is time-consuming and error-prone. This project automates the classification of resumes into job-related categories using machine learning and deep learning techniques, helping HR professionals short-list suitable candidates faster and more accurately.

---

## 📁 Dataset

* **Name:** `resume_dataset.csv`
* **Columns:**

  * `Resume`: Raw resume text.
  * `Category`: Job category label (e.g., Data Science, HR, DevOps, etc.).

> 📌 Note: Dataset is not publicly shared due to licensing. Replace `/content/drive/...` with your local or cloud path as needed.

---

## 🔧 Project Structure

```
├── preprocessing/
│   └── text_cleaning.py
├── models/
│   ├── random_forest.py
│   ├── naive_bayes.py
│   ├── lstm_model.py
│   ├── bert_model.py
│   └── distilbert_model.py
├── utils/
│   └── metrics.py
├── data/
│   └── resume_dataset.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ Technologies & Libraries

* **NLP**: NLTK, WordCloud, re
* **Traditional ML**: Scikit-learn (Random Forest, Naive Bayes, KNN)
* **Deep Learning**: TensorFlow, Keras
* **Transformers**: Hugging Face (`transformers`, `datasets`)
* **Visualization**: Matplotlib, Seaborn
* **Language**: Python 3.x

---

## 📊 Models Implemented

### ✅ 1. Traditional Machine Learning

* TF-IDF Vectorization
* Models:

  * Random Forest
  * Naive Bayes
* Achieved baseline accuracy \~85–90%

### ✅ 2. Deep Learning – LSTM

* Tokenization + Padding
* LSTM architecture with dropout layers
* Class weights for imbalance handling
* Achieved accuracy up to \~92%

### ✅ 3. Transformer-based Models

* **BERT** (`bert-base-uncased`)
* **DistilBERT** (faster training)
* Tokenization using HuggingFace `Tokenizer`
* Fine-tuned using HuggingFace `Trainer` API
* Best performance with \~94–96% accuracy

---

## 📈 Evaluation Metrics

* Accuracy Score
* Classification Report (Precision, Recall, F1-Score)
* Evaluation done using `sklearn.metrics`

---

## 📌 How to Run

### 🛠️ Install Dependencies

```bash
pip install -r requirements.txt
```

> Contents of `requirements.txt`:

```txt
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
wordcloud
tensorflow
transformers
datasets
```

### ▶️ Run Models

#### 1. Traditional ML

```bash
python models/random_forest.py
```

#### 2. LSTM Model

```bash
python models/lstm_model.py
```

#### 3. BERT / DistilBERT

```bash
python models/bert_model.py
# or
python models/distilbert_model.py
```

---

## 📦 Model Saving & Deployment

* BERT model saved at:
  `/content/drive/MyDrive/bert_resume_classifier/`

* Can be exported to TensorFlow Lite or ONNX for integration in production systems.

---

## 📉 Visualizations

* WordCloud for common keywords across resumes
* Category distribution using `seaborn` countplot
* Confusion matrices and classification metrics

---

## 🧠 Future Enhancements

* Resume ranking based on JD match %
* Integration with ATS (Applicant Tracking Systems)
* Deploy via Flask or FastAPI for real-time prediction
