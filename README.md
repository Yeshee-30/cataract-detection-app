# 👁️ Cataract Detection App

A deep-learning powered web application that detects **cataract** from eye images using a custom **Mini-ResNet** model.  
Built using **TensorFlow** and **Streamlit**.

---

## 🚀 Live Demo
👉 Add the link to your Streamlit Cloud or HuggingFace Spaces deployment here.

---

## 📌 Project Overview

This project uses a custom Mini-ResNet convolutional neural network to classify eye images as:

- **Cataract**
- **Normal**

The model was trained on a curated dataset in Google Colab and integrated into an interactive Streamlit application where users can upload an image and instantly receive a prediction along with confidence.

---

## 🧠 Model Performance

### 📊 Accuracy Curve
![Accuracy](result_screenshots/Model_Accuracy/Graph_Accuracy.png)
![Accuracy](result_screenshots/Model_Accuracy/Accuracy_Report.png)
### 📊 Confusion Matrix
![Accuracy](result_screenshots/Model_Accuracy/Confusion_Matrix.png)



---

## 🧠 Model Details

- Architecture: **Custom Mini-ResNet**
- Input Size: **256 × 256 × 3**
- Loss Function: **Binary Crossentropy**
- Optimizer: **Adam**
- Saved Model Format: **`.h5`**

Training code is provided inside the `src/` folder.

---

## 📝 Requirements

```
streamlit==1.24.0
tensorflow==2.12.0
numpy==1.23.5
pillow==10.2.0
matplotlib==3.7.1
scikit-learn==1.2.2
seaborn==0.12.2
```

These libraries are used for model training, preprocessing, and running the web app.

---

## 📸 App Screenshots

### 🏠 Home Page

![Home](result_screenshots/UI/Frontend_page.png)

### 🔍 Prediction Result

![Result](result_screenshots/UI/cataract_prediction.png)
![Result](result_screenshots/UI/Normal_Prediction.png)

---

## ✨ Author
**Yeshee Agarwal**

Feel free to reach out for collaboration or suggestions!


