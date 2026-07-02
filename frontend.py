import streamlit as st
import pickle
import fitz  # PyMuPDF
import tempfile
import re

from PIL import Image
from ocr_utils import extract_text
from scipy.sparse import hstack

# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# -----------------------------
# BLOOD RULES
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

# -----------------------------
# CLEAN TEXT
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("Blood Request Form Verification System")

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:

    file_type = uploaded_file.type

    extracted_text = ""

    # -----------------------------
    # IMAGE FILE
    # -----------------------------

    if "image" in file_type:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

            tmp.write(uploaded_file.read())

            image_path = tmp.name

        extracted_text = extract_text(image_path)

        st.image(image_path, caption="Uploaded Image")

    # -----------------------------
    # PDF FILE
    # -----------------------------

    elif "pdf" in file_type:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

            tmp.write(uploaded_file.read())

            pdf_path = tmp.name

        pdf_document = fitz.open(pdf_path)

        for page in pdf_document:

            pix = page.get_pixmap()

            image_path = "temp_page.png"

            pix.save(image_path)

            extracted_text += extract_text(image_path)

    # -----------------------------
    # TEXT PROCESSING
    # -----------------------------

    cleaned_text = clean_text(extracted_text)

    rule_score = add_rule_score(cleaned_text)

    X_text = vectorizer.transform([cleaned_text])

    X = hstack([
        X_text,
        [[rule_score]]
    ])

    prediction = model.predict(X)[0]

    # -----------------------------
    # OUTPUT
    # -----------------------------

    st.subheader("Prediction Result")

    if prediction == "valid":

        st.success("VALID BLOOD REQUEST FORM")

    else:

        st.error("INVALID DOCUMENT")

    # -----------------------------
    # OCR TEXT
    # -----------------------------

    st.subheader("Extracted OCR Text")

    st.write(extracted_text)