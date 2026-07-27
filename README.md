# ❤️ Heart Disease Prediction using Machine Learning

A Machine Learning-based web application that predicts the likelihood of heart disease based on patient health parameters. The project uses **Logistic Regression** for prediction and **Streamlit** for an interactive web interface.

## 🚀 Live Demo

🔗 **Deployed App:** https://heart-disease-prediction-aiml-58.streamlit.app/


---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help healthcare professionals identify high-risk patients and take preventive measures.

This project uses Machine Learning techniques to analyze patient health data and predict whether a person is at risk of heart disease.

---

## ✨ Features

- Predicts heart disease risk in real-time
- Interactive Streamlit web application
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Logistic Regression model
- Model evaluation using multiple performance metrics
- Probability score for predictions
- Clean and user-friendly interface

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── Dataset/
│   └── heart.csv
│
├── Images/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│
├── Model/
│   ├── heart_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project uses the **Heart Disease Dataset** containing medical information such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- Slope
- Number of Major Vessels
- Thalassemia

**Target Variable**

- 0 → No Heart Disease
- 1 → Heart Disease

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Remove Duplicate Records
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. One-Hot Encoding
7. Feature Scaling
8. Train-Test Split
9. Logistic Regression Model Training
10. Model Evaluation
11. Save Trained Model
12. Deploy with Streamlit

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 📷 Screenshots

### Home Page

Add screenshot here.

### Prediction Result

Add screenshot here.

### Confusion Matrix

Add screenshot here.

### ROC Curve

Add screenshot here.

### Feature Importance

Add screenshot here.

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/Heart-Disease-Prediction.git
```

Move into the project folder:

```bash
cd Heart-Disease-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

---

## 📌 Future Improvements

- Support additional machine learning algorithms
- Improve prediction accuracy
- Add user authentication
- Connect with cloud database
- Generate downloadable health reports
- Deploy using Docker

---

## 👩‍💻 Author

**Vanshika Sharma**

B.Tech CSE (Data Science)

GitHub: https://github.com/Vanshikashar

LinkedIn: www.linkedin.com/in/vanshikapandit

---

## 📄 License

This project is created for educational and learning purposes.