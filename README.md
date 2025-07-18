# 🧾 Invoice Extraction & Classification System

**Automated Invoice Processing using OpenCV, OCR, and Machine Learning**

---

## **Features**

- **OCR Field Extraction** – Extracts **Vendor**, **Invoice No**, **Date**, and **Total Amount** from invoice images.  
- **Invoice Classification** – Classifies invoices as **Valid**, **Duplicate**, or **Suspicious**.  
- **Robust to OCR Errors** – Handles variations like **"Vnvoice No"**, **"Inv No"**, etc.  
- **Interactive Web Interface** – Built with **Streamlit**.

---

## **Tech Stack**

- **OpenCV** – Image preprocessing  
- **Tesseract OCR** – Text extraction  
- **scikit-learn** – ML Classification (RandomForest)  
- **Streamlit** – Web interface  

---

## **Setup Instructions**

### install python dependencies

```bash
pip install -r requirements.txt

### 2 Install Tesseract OCR

### 3 Run the app
streamlit run app.py

## **Invoice Labels**
Label	|   Meaning
Valid	 |  Normal invoice
Duplicate | Invoice number ends with 00
Suspicious | Total amount is 0


## **Project Structure**
invoice-extraction/
├── app.py                 # Streamlit App
├── model/classifier.pkl   # Trained ML Model
├── synthetic_invoices/    # Images + Dataset
├── requirements.txt

## **Author**
**Shahnawaz Hussain**
