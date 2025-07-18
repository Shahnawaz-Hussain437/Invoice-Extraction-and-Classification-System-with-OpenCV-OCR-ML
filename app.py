import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import pickle

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# Load trained classifier
with open("model/classifier.pkl", "rb") as f:
    clf = pickle.load(f)

st.title("🧾 Invoice Extraction & Classification System (OCR + OpenCV + ML)")

uploaded_file = st.file_uploader("Upload Invoice Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Invoice", use_column_width=True)

    # Convert to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR with pytesseract
    text = pytesseract.image_to_string(thresh)

    st.subheader("🔍 OCR Extracted Text")
    st.text(text)

    # Extract fields using flexible regex (handles OCR errors)

    # Vendor
    vendor_match = re.search(r'Vendor[:\s]*(.*)', text, re.IGNORECASE)
    vendor = vendor_match.group(1).strip() if vendor_match else "Unknown"

    # Invoice Number (handles "Invoice", "Vnvoice", "Inv")
    invoice_match = re.search(r'(Invoice|Vnvoice|Inv)\s*No[:\s]*([0-9]+)', text, re.IGNORECASE)
    invoice_num = invoice_match.group(2).strip() if invoice_match else "0"

    # Date
    date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
    date = date_match.group(1) if date_match else "Not Found"

    # Total Amount (handles flexible spacing)
    amount_match = re.search(r'Total\s*Amount[:\s]*([\d\.]+)', text, re.IGNORECASE)
    total_amount = float(amount_match.group(1)) if amount_match else 0.0

    # Display extracted fields
    st.subheader("📝 Extracted Fields")
    st.write(f"**Vendor:** {vendor}")
    st.write(f"**Invoice Number:** {invoice_num}")
    st.write(f"**Date:** {date}")
    st.write(f"**Total Amount:** {total_amount}")

    # Prepare features for classifier
    features = [[len(vendor), len(invoice_num), total_amount]]

    # Predict
    result = clf.predict(features)[0]

    st.subheader("📊 Classification Result")
    st.write(f"**{result}**")
