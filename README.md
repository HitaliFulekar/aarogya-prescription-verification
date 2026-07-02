# Aarogya: ML-Based Blood Request Form Verification System

An AI-powered document verification system that automatically validates uploaded blood request forms using OCR and Machine Learning.

---

## Problem Statement

Blood donation platforms often require users to upload blood request forms. Manual verification is time-consuming and prone to errors. This project automates the verification process by classifying uploaded documents as **Valid** or **Invalid** blood request forms.

---

## Features

- Upload Image or PDF
- OCR-based text extraction using EasyOCR
- Text preprocessing
- TF-IDF feature extraction
- Linear SVM classification
- Rule-based medical keyword verification
- Streamlit web interface
- Real-time document prediction

---

## Tech Stack

- Python
- Streamlit
- EasyOCR
- Scikit-learn
- TF-IDF
- Linear SVM
- OpenCV
- PyMuPDF
- Pandas
- SciPy

---

## Workflow

1. Upload Image/PDF
2. OCR extracts text
3. Text preprocessing
4. TF-IDF vectorization
5. Rule-based keyword scoring
6. Linear SVM prediction
7. Display verification result

---

## Model Performance

- **Accuracy:** 94.52%
- OCR: EasyOCR
- Feature Extraction: TF-IDF
- Classifier: Linear SVM

---

## Project Structure

```
aarogya-prescription-verification
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── frontend.py
├── app.py
├── ocr_utils.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/HitaliFulekar/aarogya-prescription-verification.git

cd aarogya-prescription-verification

pip install -r requirements.txt

streamlit run frontend.py
```

---

## Future Improvements

- Deep Learning-based Document Classification
- LayoutLM Integration
- Named Entity Recognition (NER)
- Multi-language OCR Support
- Confidence Score Visualization

---

## Developer

**Hitali Fulekar**

B.Tech CSE

Dr. B. R. Ambedkar National Institute of Technology Jalandhar

---

## If you found this project useful, consider giving it a star!