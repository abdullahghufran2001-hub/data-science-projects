# 📊 Universal Data Analytics Dashboard

### Upload Any CSV or Excel File • Analyze • Visualize • Download Insights

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🚀 Overview

**Universal Data Analytics Dashboard** is an interactive analytics application built with **Python and Streamlit**.

The dashboard allows users to upload a **CSV or Excel file** and instantly explore the dataset through:

- 📊 Dataset KPIs
- 📋 Column information
- 🔎 Dataset preview
- 📈 Statistical summary
- 📊 Numeric analysis
- 📌 Category analysis
- 🔥 Correlation analysis
- ⚠️ Missing-value analysis
- 📥 Downloadable summary report

The goal of the project is to provide a simple and reusable analytics interface where users can upload different datasets without building a separate dashboard for every dataset.

---

# 🎯 Project Objective

Traditional data analysis often requires writing separate code for every dataset.

This project provides a **universal analytics workflow**:

```text
Upload Dataset
      ↓
Dataset Detection
      ↓
Data Overview
      ↓
Data Quality Analysis
      ↓
Statistical Analysis
      ↓
Numeric Analysis
      ↓
Category Analysis
      ↓
Correlation Analysis
      ↓
Missing Value Report
      ↓
Download Summary
```

---

# ✨ Key Features

## 🔐 1. Login System

The application starts with a login screen.

Users must provide:

```text
Username
Password
```

Default credentials configured in the application:

```text
Username: admin
Password: admin123
```

After successful authentication, the analytics dashboard becomes available.

> **Note:** The credentials in this demo are hardcoded for demonstration purposes. Production applications should use secure authentication and environment variables.

---

# 📂 2. Universal File Upload

The dashboard supports:

```text
CSV
Excel (.xlsx)
```

Users can upload their dataset directly through the Streamlit file uploader.

Example:

```text
📂 Upload CSV or Excel File
```

The application automatically determines how to read the uploaded file.

---

# 📊 3. Dataset Overview

After uploading a dataset, the dashboard calculates four important KPIs:

| KPI | Description |
|---|---|
| 📄 Rows | Total number of records |
| 📊 Columns | Total number of columns |
| ⚠️ Missing | Total missing values |
| 🔁 Duplicates | Total duplicate rows |

Example:

```text
┌────────────┬────────────┬────────────┬────────────┐
│    Rows    │  Columns   │  Missing   │ Duplicates │
├────────────┼────────────┼────────────┼────────────┤
│   10,000   │     12     │     25     │      8     │
└────────────┴────────────┴────────────┴────────────┘
```

---

# 📋 4. Dataset Information

The application automatically generates a column-level information table containing:

- Column name
- Data type
- Missing values

Example:

| Column | Data Type | Missing Values |
|---|---|---:|
| Age | int64 | 0 |
| Salary | float64 | 5 |
| Department | object | 2 |

This helps users understand the structure and quality of their dataset.

---

# 📄 5. Dataset Preview

The complete uploaded dataset is displayed directly inside the dashboard.

This allows users to:

- Inspect records
- Understand columns
- Identify unusual values
- Verify uploaded data

---

# 📈 6. Statistical Summary

The application generates an automatic statistical summary using Pandas.

It can provide information such as:

```text
Count
Mean
Standard Deviation
Minimum
25%
50%
75%
Maximum
```

The summary adapts to the columns available in the uploaded dataset.

---

# 🔢 7. Automatic Numeric Column Detection

The application automatically detects numerical columns.

For example:

```text
Age
Salary
Sales
Quantity
Price
Marks
Revenue
```

Users can select a numeric column for deeper analysis.

---

# 📊 8. Numeric Analysis

For the selected numeric column, the dashboard generates two interactive Plotly visualizations.

### Histogram

Used to understand the distribution of numerical values.

```text
Distribution
      ↑
      │       ███
      │   ███████
      │ ███████████
      └────────────────→ Values
```

### Box Plot

Used to understand:

- Distribution
- Median
- Spread
- Potential outliers

---

# 📌 9. Automatic Category Detection

The application automatically detects categorical columns such as:

```text
Gender
City
Department
Category
Status
Country
Product
```

Users can select a categorical column for analysis.

---

# 📊 10. Category Analysis

The dashboard identifies the **top 10 categories** based on frequency.

Two visualizations are generated:

### Bar Chart

Shows the number of records belonging to each top category.

### Pie Chart

Shows the distribution of the selected categories.

---

# 🔥 11. Correlation Heatmap

When the dataset contains more than one numerical column, the dashboard automatically calculates correlations.

Example:

```text
              Sales   Profit   Quantity
Sales          1.00     0.82      0.54
Profit         0.82     1.00      0.41
Quantity       0.54     0.41      1.00
```

The interactive heatmap helps identify relationships between numerical variables.

---

# ⚠️ 12. Missing Values Report

The dashboard generates a dedicated missing-value report.

Example:

| Column | Missing Values |
|---|---:|
| Age | 0 |
| Salary | 5 |
| Department | 2 |
| Revenue | 0 |

This makes it easier to identify columns requiring data cleaning.

---

# 📥 13. Download Summary Report

Users can download an automatically generated CSV summary containing:

```text
Rows
Columns
Missing Values
Duplicate Rows
```

The generated file is:

```text
dataset_summary.csv
```

This makes the dashboard useful for quick dataset profiling and reporting.

---

# 🎨 User Interface

The application includes a custom interface with:

- 🔐 Login screen
- 📊 Dashboard header
- 🎨 Custom CSS
- 📂 File uploader
- 📈 KPI cards
- 📋 Interactive data tables
- 📊 Plotly charts
- 📥 Download button
- 🚪 Logout functionality

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Web dashboard |
| Pandas | Data processing |
| Plotly Express | Interactive charts |
| Plotly Figure Factory | Correlation heatmap |
| CSV | Dataset input |
| Excel | Dataset input |

---

# 📁 Project Structure

```text
Universal-Data-Analytics-Dashboard/
│
├── app.py
│
├── data/
│   └── sample_data.csv
│
├── screenshots/
│   ├── login.png
│   ├── dashboard.png
│   ├── numeric_analysis.png
│   └── correlation.png
│
├── reports/
│   └── dataset_summary.csv
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Universal-Data-Analytics-Dashboard.git
```

Open the project:

```bash
cd Universal-Data-Analytics-Dashboard
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

# ▶️ Run the Dashboard

Use:

```bash
streamlit run app.py
```

Or:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

# 🔐 Login

For the current demo version:

```text
Username:
admin

Password:
admin123
```

After login:

```text
Login
  ↓
Upload Dataset
  ↓
Analyze Dataset
```

---

# 📦 requirements.txt

```text
streamlit
pandas
plotly
openpyxl
```

---

# 🧪 Supported Dataset Types

The dashboard is designed to work with datasets containing different types of information.

### Business Data

```text
Sales
Revenue
Profit
Products
Customers
Orders
```

### HR Data

```text
Employees
Salary
Department
Experience
Performance
```

### Student Data

```text
Students
Marks
Attendance
Study Hours
Results
```

### Customer Data

```text
Customer ID
Age
City
Purchase
Category
```

---

# 💼 Real-World Use Cases

This dashboard can be used as a starting point for:

### 🛒 E-Commerce Analytics

Analyze sales, products and customers.

### 👥 Customer Analytics

Understand customer distribution and behavior.

### 💰 Financial Analytics

Analyze revenue, transactions and financial metrics.

### 🎓 Education Analytics

Analyze student marks, attendance and performance.

### 👨‍💼 HR Analytics

Analyze employee data, departments and performance.

---

# 🧠 Data Analysis Workflow

```text
                 UPLOAD
                    ↓
             DATA PROFILING
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   NUMERIC DATA            CATEGORICAL DATA
        ↓                       ↓
 Histogram / Boxplot       Bar / Pie Chart
        │                       │
        └───────────┬───────────┘
                    ↓
             CORRELATION
                    ↓
           MISSING VALUES
                    ↓
            SUMMARY REPORT
                    ↓
               DOWNLOAD
```

---

# 🚀 Future Enhancements

The dashboard can be extended with:

## 🤖 Machine Learning

- Automatic target detection
- Classification
- Regression
- Prediction
- Model comparison
- Feature importance

## 🧹 Data Cleaning Studio

- Remove duplicates
- Fill missing values
- Rename columns
- Change data types
- Outlier treatment

## 📊 Advanced Analytics

- Interactive filters
- Date-range filtering
- KPI cards
- Pivot analysis
- Group-by analysis
- Automated insights

## 📈 Advanced Visualizations

- Scatter plots
- Time-series charts
- Treemaps
- Geographic maps
- Interactive dashboards

## 📄 Reporting

- PDF reports
- Excel reports
- Automated analysis reports
- Downloadable charts

---

# 🔒 Production Considerations

The current project is designed as a portfolio/demo application.

For production deployment, consider:

- Secure authentication
- Password hashing
- Environment variables
- User roles
- File-size limits
- Input validation
- Logging
- Error monitoring
- Database integration
- Deployment security

---

# 🎯 Project Goals

The project demonstrates how Python can be used to build a reusable analytics application that transforms:

```text
RAW DATA
   ↓
INFORMATION
   ↓
VISUALIZATION
   ↓
INSIGHTS
```

without requiring users to manually write analysis code for every uploaded dataset.

---

# 📚 Skills Demonstrated

Through this project, the following skills are demonstrated:

- Python
- Pandas
- Data Profiling
- Data Cleaning Concepts
- Exploratory Data Analysis
- Statistical Analysis
- Data Visualization
- Plotly
- Streamlit
- CSV Handling
- Excel Handling
- Dashboard Development
- GitHub Project Documentation

---

# 👨‍💻 Author

## Abdullah Ghufran

**Data Science | Data Analytics | Python | SQL | Machine Learning | Power BI**

---

# ⭐ Support the Project

If you find this project useful:

⭐ Star the repository  
🍴 Fork the repository  
🐛 Report issues  
💡 Suggest improvements  

---

# 🚀 Final Workflow

```text
Upload Any Dataset
        ↓
Understand Your Data
        ↓
Analyze Your Data
        ↓
Visualize Your Data
        ↓
Discover Relationships
        ↓
Find Data Quality Issues
        ↓
Download Your Summary
```

## 📊 Universal Analytics Dashboard

> **One Dashboard. Any Dataset. Instant Insights.**
