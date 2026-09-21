# 🎓 AI-Powered Student Performance Prediction System

### Machine Learning Based Student Performance Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

The **AI-Powered Student Performance Prediction System** is a Machine Learning project designed to analyze student academic performance and predict the expected result based on important academic and behavioral factors.

The system uses a **Random Forest Classification model** to learn patterns from historical student data and generate predictions for new students.

The project also provides:

- 📊 Model accuracy analysis
- 🧠 Feature importance analysis
- 🎓 Student performance analysis
- 🤖 AI-based prediction
- 📈 Prediction probability
- ✅ Input validation

---

# 🎯 Project Objective

The primary objective of this project is to use Machine Learning to understand the relationship between student performance factors and their final academic result.

The system analyzes:

- Attendance
- Study Hours
- Previous Marks
- Assignment Score
- Test Score

and predicts the student's expected result.

---

# 🧠 How the System Works

```text
Student Dataset
      ↓
Data Loading
      ↓
Column Validation
      ↓
Feature Selection
      ↓
Train / Test Split
      ↓
Random Forest Classifier
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Student Input
      ↓
AI Prediction
      ↓
Prediction Probability
      ↓
Performance Visualization
```

---

# 📊 Input Features

| Feature | Description |
|---|---|
| Attendance | Student attendance percentage |
| StudyHours | Number of study hours |
| PreviousMarks | Previous academic marks |
| AssignmentScore | Assignment performance |
| TestScore | Test performance |
| Result | Target variable |

The project validates the required columns before training the model.

---

# 🤖 Machine Learning Model

## Random Forest Classifier

The project uses a **Random Forest Classifier** for prediction.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

The dataset is divided into training and testing data using an **80/20 split**.

```text
80% → Training Data
20% → Testing Data
```

---

# 📏 Model Evaluation

The model performance is evaluated using:

### Accuracy

```text
Accuracy =
Correct Predictions / Total Predictions
```

The system displays the calculated model accuracy after testing the trained model.

---

# 🔮 Student Prediction

After training the model, users can enter details for a new student:

```text
Attendance
Study Hours
Previous Marks
Assignment Score
Test Score
```

The model then generates:

```text
Predicted Result
       +
Prediction Confidence
```

The prediction probability is obtained using:

```python
model.predict_proba()
```

---

# 📈 Visualizations

The project generates four major visualizations.

## 1. Model Accuracy

Displays the Random Forest model's accuracy.

```text
MODEL ACCURACY
      │
      │ █████████████
      │ █████████████
      └───────────────
       Random Forest
```

---

## 2. Feature Importance

Shows which student-related features contributed most to the model's predictions.

Features analyzed:

- Attendance
- Study Hours
- Previous Marks
- Assignment Score
- Test Score

---

## 3. Student Performance Analysis

Displays the entered student's:

- Attendance
- Previous Marks
- Assignment Score
- Test Score

This provides a quick visual overview of the student's performance.

---

## 4. AI Prediction Probability

Displays the probability associated with the model's predicted classes.

```text
Prediction
     ↓
Class Probabilities
     ↓
Confidence %
```

---

# 🛡️ Input Validation

The system validates student inputs before prediction.

### Attendance

```text
0 – 100%
```

### Study Hours

```text
0 – 20 hours
```

### Previous Marks

```text
0 – 100
```

### Assignment Score

```text
0 – 100
```

### Test Score

```text
0 – 100
```

Invalid values generate an input error instead of being passed to the model.

---

# 📁 Project Structure

```text
AI-Student-Performance-Prediction/
│
├── data/
│   └── sample_students.csv
│
├── src/
│   └── student_prediction.py
│
├── outputs/
│   ├── model_accuracy.png
│   ├── feature_importance.png
│   ├── student_performance.png
│   └── prediction_probability.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Dataset handling |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| Random Forest | Classification |
| CSV | Dataset format |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Student-Performance-Prediction.git
```

Move into the project:

```bash
cd AI-Student-Performance-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Make sure your dataset is available in the project data directory.

Then run:

```bash
python src/student_prediction.py
```

---

# 📋 Dataset Format

The CSV should contain the following columns:

```text
Attendance
StudyHours
PreviousMarks
AssignmentScore
TestScore
Result
```

Example:

| Attendance | StudyHours | PreviousMarks | AssignmentScore | TestScore | Result |
|---:|---:|---:|---:|---:|---|
| 85 | 5 | 78 | 82 | 80 | Pass |
| 60 | 2 | 55 | 58 | 52 | Fail |
| 92 | 7 | 88 | 90 | 91 | Pass |

---

# 💡 Business / Educational Use Case

This type of system can be used as a foundation for an educational analytics platform.

Potential applications include:

- Early identification of students needing support
- Academic performance monitoring
- Student analytics dashboards
- Performance trend analysis
- Personalized academic interventions
- Institutional performance reporting

**Important:** A model prediction should be treated as analytical support, not as the sole basis for decisions about a student.

---

# 🚀 Future Improvements

The project can be extended with:

### 📊 Advanced Analytics

- Student performance dashboard
- Correlation analysis
- Attendance trends
- Performance distributions

### 🤖 Advanced Machine Learning

- Logistic Regression
- Decision Tree
- XGBoost
- Gradient Boosting
- Hyperparameter tuning
- Cross-validation

### 📈 Advanced Evaluation

- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

### 🖥️ Application

- Streamlit web application
- Interactive student prediction form
- Dashboard
- Batch prediction through CSV

### 💾 Model Management

- Save trained model
- Load trained model
- Prediction API
- Model versioning

---

# 🔬 Machine Learning Pipeline

```text
Historical Student Data
          ↓
     Data Validation
          ↓
     Feature Selection
          ↓
      Train / Test
          ↓
    Random Forest
          ↓
      Evaluation
          ↓
     New Student
          ↓
       Prediction
          ↓
     Probability
          ↓
    Visualization
```

---

# 📌 Key Features

- ✅ CSV dataset support
- ✅ Required-column validation
- ✅ Random Forest classification
- ✅ 80/20 train-test split
- ✅ Accuracy calculation
- ✅ Student input system
- ✅ Input validation
- ✅ Prediction generation
- ✅ Prediction probability
- ✅ Feature importance
- ✅ Student performance visualization
- ✅ Model accuracy visualization

---

# 🎓 Learning Outcomes

Through this project, the following Data Science concepts are demonstrated:

```text
Python
   ↓
Pandas
   ↓
Data Preparation
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Machine Learning
   ↓
Random Forest
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Data Visualization
```

---

# 👨‍💻 Author

## Abdullah Ghufran

**Aspiring Data Scientist**

Skills:

`Python` `Pandas` `NumPy` `SQL` `Machine Learning` `Power BI` `Data Visualization`

---

# ⭐ Project

If you find this project useful, consider giving the repository a ⭐.

---

## 🚀 From Data to Intelligence

> **Analyze the data. Discover the pattern. Build the model. Generate the insight.**
