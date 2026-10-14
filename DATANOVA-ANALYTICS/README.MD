# 📊 Universal Analytics Pro — DataNova Analytics

### An Interactive Data Analytics, Visualization & Machine Learning Platform

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

**Universal Analytics Pro** is an interactive analytics platform built with Python and Streamlit. It helps users explore datasets, clean data, visualize patterns, evaluate data quality, train machine learning models, and generate predictions from uploaded CSV or Excel files.

The platform also includes a **Future Salary & Company Growth Forecaster** that calculates scenario-based projections using a user-defined annual growth rate.

## ✨ Key Features

### 🏠 1. Executive Dashboard & Overview
- Automatic dataset summary and key performance indicators (KPIs).
- Record count, column count, missing values, and duplicate-row metrics.
- Automatic findings for numeric and categorical columns.
- Dataset structure and column-completeness charts.
- Monthly trend analysis when a date column is detected.
- Category performance rankings.
- Numeric distributions, box plots, and outlier analysis.
- Correlation heatmap for numeric columns.

### 📊 2. Interactive Visual Analytics
Create interactive visualizations using Plotly:
- Histograms
- Box plots
- Violin plots
- Bar charts
- Pie charts
- Donut charts
- Treemaps
- Scatter plots
- Pearson correlation analysis

### 🧹 3. Data Cleaning & Quality
- Upload CSV and XLSX datasets.
- Optional duplicate-row removal.
- Missing-value handling:
  - Keep missing values unchanged
  - Drop rows with missing values
  - Fill numeric missing values with median
  - Fill numeric missing values with mean
  - Fill missing values with zero
  - Fill missing values with `Not Available`
- Data types, missing-value counts, missing percentages, and unique-value counts.
- IQR-based potential outlier detection.
- Downloadable data-quality report.

### 🎛️ 4. Interactive Filters & Data Explorer
- Filter records by categorical values.
- Filter data using date ranges.
- Filter numeric columns using a range slider.
- Search across all columns.
- Explore matching records in an interactive table.

### 🤖 5. Dataset-Based Machine Learning

Train a model using a selected target column and input features.

**Supported prediction tasks**
- **Regression:** Predict numeric values such as sales, revenue, or salary when suitable data is available.
- **Classification:** Predict categories such as approval status or customer outcomes.
- **Auto Detect:** Select a task based on the target column's type and number of unique values.

**Supported algorithms**
- Random Forest
- Extra Trees
- Gradient Boosting
- Linear Regression for regression tasks

**Training configuration**
- Select target and input features.
- Configure test-set percentage.
- Set the number of estimators/trees for supported ensemble algorithms.
- Configure maximum tree depth.
- Configure minimum samples per leaf.

**Model evaluation**
- Regression: MAE, RMSE, and R² score.
- Classification: Accuracy, weighted F1, confusion matrix, and classification report.
- Feature-importance visualization for supported tree-based models.
- Enter a new record to generate a prediction.
- Export supported prediction results.

### 🔮 6. Future Salary & Company Growth Forecaster

Explore possible future values over a 1–10 year period.

- Salary growth scenarios.
- Company revenue/business-value scenarios.
- User-defined current value.
- User-defined annual growth rate.
- Conservative, base, and optimistic scenarios.
- Interactive multi-scenario line chart.
- Year-by-year projection table.
- Download forecast data as CSV.

**Formula used**

`Future Value = Current Value × (1 + Annual Growth Rate / 100) ^ Number of Years`

This is a compound-growth scenario calculator, not a trained forecasting model. Results depend on the growth rate entered by the user and are not guaranteed future outcomes.

### 📥 7. Export Center
- Export prepared data.
- Download data-quality reports.
- Download future-growth projections.
- Export supported prediction results.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data loading, cleaning, and analysis |
| NumPy | Numerical operations |
| Plotly Express | Interactive visualizations |
| Scikit-learn | Machine learning and evaluation |
| OpenPyXL | Excel file support |

---

## 📂 Project Structure

```text
Universal-Analytics-Pro/
│
├── Universal_Analytics_Pro_Enhanced.py
├── README.md
└── requirements.txt
```

Make sure the Python filename matches the actual filename in your GitHub repository.

## ⚙️ Installation & Setup

### 1. Clone the repository

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your GitHub username and repository name.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create a virtual environment (recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file with the following contents:

```text
streamlit
pandas
numpy
plotly
scikit-learn
openpyxl
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run Universal_Analytics_Pro_Enhanced.py
```

If your Python file has a different name, replace the filename in the command.

### 5. Login credentials

The current code includes demo credentials:

```text
Username: admin
Password: admin123
```

**Security note:** These credentials are hard-coded for demonstration. Replace them with secure authentication before deploying the application publicly.

### 6. Upload a dataset

1. Log in to the application.
2. Upload a CSV or XLSX file.
3. Choose cleaning options and filters.
4. Explore the Overview and Visual Analytics tabs.
5. Open Prediction Lab to configure and train a model.
6. Download supported reports and prediction outputs.

---

## 🧠 How the Prediction Lab Works

1. **Choose a target:** Select the column the model should predict.
2. **Choose input features:** Select columns that contain information available before the outcome is known. Do not include the target itself as an input.
3. **Choose the task and algorithm:** Use regression for numeric targets or classification for categorical targets.
4. **Configure training:** Set the test split and model parameters.
5. **Train and evaluate:** The application splits the data into training and test sets, fits a preprocessing/model pipeline, and reports evaluation metrics.
6. **Predict new records:** Enter values for the selected input features and generate an estimate.

### Important ML considerations
- Model performance depends on dataset size, quality, target selection, and feature relevance.
- Avoid target leakage and inappropriate identifier columns.
- A single train/test split can be unstable on small datasets.
- Predictions are estimates, not guarantees or professional decisions.

---

## 📈 Example Use Cases

- Analyze sales and revenue datasets.
- Explore customer and product distributions.
- Inspect data completeness and potential outliers.
- Predict numeric business metrics from suitable datasets.
- Classify outcomes using labeled data.
- Compare assumed salary or company-growth scenarios.
- Create interactive reports from CSV or Excel files.

These are potential use cases. Predictions are meaningful only when the uploaded dataset contains suitable features and target values.

## 🚀 Future Improvements

- Time-series forecasting using historical date-based data.
- Cross-validation and hyperparameter tuning.
- Additional model comparison and explainability tools.
- Saved model artifacts and prediction history.
- Secure authentication and deployment improvements.
- Automated report generation.

## 🔐 Privacy & Security

- Upload only datasets you are authorized to use.
- The demo login is not production-grade authentication.
- Review hosting settings before uploading confidential or personal data.
- Do not publish secrets, credentials, or sensitive datasets in the repository.

## 👨‍💻 Author

**Abdullah Ghufran**

Data Analytics | Python | Machine Learning | Interactive Dashboards

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

*Built with Python, Streamlit, Pandas, Plotly, and Scikit-learn.*

