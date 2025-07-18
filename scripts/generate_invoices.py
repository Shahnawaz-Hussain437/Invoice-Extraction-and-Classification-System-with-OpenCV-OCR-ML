import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import random
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

fake = Faker()

# Create output folders
os.makedirs("synthetic_invoices/images", exist_ok=True)
os.makedirs("model", exist_ok=True)

data = []

# Load a font (use system font or a ttf you have)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

for i in range(500):
    # Create blank white image
    img = Image.new('RGB', (600, 400), color = (255,255,255))
    draw = ImageDraw.Draw(img)

    # Generate random invoice data
    vendor = fake.company()
    invoice_number = str(fake.random_int(100000,999999))
    date = fake.date(pattern="%d/%m/%Y")
    total_amount = round(random.uniform(100, 2000), 2)

    # Draw invoice fields
    draw.text((50, 50), f"Vendor: {vendor}", fill=(0,0,0), font=font)
    draw.text((50, 100), f"Invoice No: {invoice_number}", fill=(0,0,0), font=font)
    draw.text((50, 150), f"Date: {date}", fill=(0,0,0), font=font)
    draw.text((50, 200), f"Total Amount: {total_amount}", fill=(0,0,0), font=font)

    # Convert to OpenCV image for saving
    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Save image
    img_name = f"invoice_{i}.png"
    cv2.imwrite(f"synthetic_invoices/images/{img_name}", cv_img)

    # Generate labels
    if total_amount == 0:
        label = "Suspicious"
    elif invoice_number.endswith("00"):
        label = "Duplicate"
    else:
        label = "Valid"

    # Append features
    data.append([
        len(vendor),
        len(invoice_number),
        total_amount,
        label
    ])

# Save annotations CSV
df = pd.DataFrame(data, columns=["vendor_len", "inv_num_len", "amount", "label"])
df.to_csv("synthetic_invoices/annotations.csv", index=False)

print("✅ Generated 500 synthetic invoices and annotations.csv")

# Train classifier
X = df[["vendor_len", "inv_num_len", "amount"]]
y = df["label"]

clf = RandomForestClassifier()
clf.fit(X, y)

# Save model
with open("model/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Trained classifier and saved to model/classifier.pkl")
