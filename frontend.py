import streamlit as st
import pickle
import fitz
import tempfile
import re

from ocr_utils import extract_text
from scipy.sparse import hstack

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="Aarogya - Blood Request Verification",
    page_icon="🩸",
    layout="centered"
)

# ------------------------------------
# LOAD MODEL
# ------------------------------------

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ------------------------------------
# BLOOD KEYWORDS
# ------------------------------------

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

# ------------------------------------
# RULE SCORE
# ------------------------------------

def add_rule_score(text):

    score = 0
    text = str(text).lower()

    for word in blood_keywords:
        if word in text:
            score += 3

    return score

# ------------------------------------
# CLEAN TEXT
# ------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

# ------------------------------------
# SIDEBAR
# ------------------------------------

with st.sidebar:

    st.title("🩸 Aarogya")

    st.markdown("### ML-Based Prescription Verification")

    st.write(
        """
This application verifies uploaded blood request forms using:

-> EasyOCR

-> TF-IDF Feature Extraction

-> Linear SVM Classifier

-> Rule-Based Validation
"""
    )

    st.divider()

    st.write("Developed by")

    st.write("**Hitali Fulekar**")

    st.write("NIT Jalandhar")

# ------------------------------------
# TITLE
# ------------------------------------

st.title("🩸 Blood Request Form Verification")

st.write(
    """
Upload a **Blood Request Form** in **Image** or **PDF** format.

The system automatically extracts text using OCR and predicts whether the uploaded document is a **Valid Blood Request Form** or an **Invalid Document**.
"""
)

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "pdf"]
)

# ------------------------------------
# PROCESS FILE
# ------------------------------------

if uploaded_file is not None:

    with st.spinner("Processing document..."):

        file_type = uploaded_file.type

        extracted_text = ""

        # IMAGE

        if "image" in file_type:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

                tmp.write(uploaded_file.read())

                image_path = tmp.name

            st.image(
                image_path,
                caption="Uploaded Image",
                use_container_width=True
            )

            extracted_text = extract_text(image_path)

        # PDF

        elif "pdf" in file_type:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

                tmp.write(uploaded_file.read())

                pdf_path = tmp.name

            pdf_document = fitz.open(pdf_path)

            for page in pdf_document:

                pix = page.get_pixmap()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:

                    image_path = temp_img.name

                pix.save(image_path)

                extracted_text += extract_text(image_path)

        # ------------------------------------
        # ML Prediction
        # ------------------------------------

        cleaned_text = clean_text(extracted_text)

        rule_score = add_rule_score(cleaned_text)

        X_text = vectorizer.transform([cleaned_text])

        X = hstack([
            X_text,
            [[rule_score]]
        ])

        prediction = model.predict(X)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == "valid":

        st.success("✅ VALID BLOOD REQUEST FORM")

    else:

        st.error("❌ INVALID DOCUMENT")

    st.divider()

    with st.expander("📄 View Extracted OCR Text"):

        st.write(extracted_text)