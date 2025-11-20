# Handwritten Text Recognition App ✍️📄  
A simple, fast, and user-friendly Streamlit application that converts handwritten text into digital text using deep learning.

This project allows you to upload an image of handwritten notes and instantly extract readable text.  
The app runs completely in the browser via Streamlit — no setup required.

---

## 🚀 Features

- 📝 Convert handwritten text images into digital text  
- 📤 Upload JPG/PNG images  
- ⚡ Fast and lightweight OCR  
- 🎯 High accuracy on clean handwriting  
- 🌐 Works online through Streamlit Cloud  
- 📱 Clean and simple interface  

---

## 🧠 How It Works

The app uses a pretrained OCR model that processes the input image in two steps:

1. **Text Detection** – Finds the regions containing text  
2. **Text Recognition** – Predicts characters inside each region  

Both steps are combined to produce the final extracted text.

---

## 🖼️ Usage

### 1. Upload your handwritten note image  
Supported formats: **JPG, JPEG, PNG**

### 2. The model analyzes your image  
Extraction may take a few seconds depending on size.

### 3. Get clean digital text  
You can copy the output or reuse it anywhere.

---

## 🛠️ Installation (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
