import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("dataset.csv")

# remove empty rows
df = df.dropna()

# -----------------------------
# TEXT CLEANING
# -----------------------------

# convert to lowercase
df["text"] = df["text"].astype(str).str.lower()

# remove special characters
df["text"] = df["text"].apply(
    lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x)
)

# remove extra spaces
df["text"] = df["text"].apply(
    lambda x: re.sub(r'\s+', ' ', x)
)

# -----------------------------
# RULE-BASED FEATURES
# -----------------------------

blood_keywords = [
    "blood requisition",
    "blood request",
    "blood bank",
    "cross match",
    "crossmatch",
    "component required",
    "units required",
    "blood component",
    "packed red cells",
    "prbc",
    "platelets",
    "ffp",
    "cryoprecipitate",
    "transfusion",
    "transfusion history",
    "blood group",
    "ward bed",
    "issued by blood bank",
    "blood bank use only"
]

def add_rule_score(text):

    score = 0
    text = str(text).lower()

    for word in blood_keywords:
        if word in text:
            score += 3

    return score

df["rule_score"] = df["text"].apply(add_rule_score)

# -----------------------------
# TF-IDF FEATURES
# -----------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9
)

X_text = vectorizer.fit_transform(df["text"])

# -----------------------------
# COMBINE FEATURES
# -----------------------------

X = hstack([
    X_text,
    df["rule_score"].values.reshape(-1, 1)
])

y = df["label"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# MODEL
# -----------------------------

model = LinearSVC(
    class_weight='balanced'
)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nMODEL SAVED SUCCESSFULLY")