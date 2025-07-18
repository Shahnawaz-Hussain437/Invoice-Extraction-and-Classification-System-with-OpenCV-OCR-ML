🧾 Invoice Extraction & Classification System
Automated Invoice Processing using OpenCV, OCR, and Machine Learning

Features
OCR Field Extraction – Extracts Vendor, Invoice No, Date, and Total Amount from invoice images.

Invoice Classification – Classifies invoices as Valid, Duplicate, or Suspicious.

Robust to OCR Errors – Handles text variations like "Vnvoice No" vs "Invoice No".

Interactive Web Interface – Built with Streamlit.

Tech Stack
OpenCV – Image preprocessing

Tesseract OCR – Text extraction

scikit-learn – ML Classification (RandomForest)

Streamlit – Web interface

Setup Instructions

1️⃣ Install Python Dependencies

pip install -r requirements.txt

2️⃣ Install Tesseract OCR

Download from: Tesseract Windows Installer

Default path used in app.py:
C:\Program Files (x86)\Tesseract-OCR\tesseract.exe

first run generate_invoices.py to generate synthetic data to train the classifier

3️⃣ Run the App

streamlit run app.py

Invoice Labels
Label	      Description
Valid	      Normal invoice
Duplicate	  Invoice number ends with 00
Suspicious	  Total amount is 0


Project Structure

invoice-extraction/
├── app.py                 # Streamlit App
├── model/classifier.pkl   # Trained Model
├── synthetic_invoices/    # Images + Dataset
├── requirements.txt

Author
Shahnawaz Hussain
LinkedIn