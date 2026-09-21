# 🎬 StreamFlix — Data Science & AI Analytics Platform

### End-to-End Streaming Analytics, Machine Learning & Recommendation System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Analytics-3F4F75)
![Status](https://img.shields.io/badge/Project-Completed-success)

</p>

---

# 📌 Project Overview

**StreamFlix** is an end-to-end **Data Science and Artificial Intelligence analytics project** designed around a streaming-platform business model.

The project combines:

- 📊 Data Analysis
- 🧹 Data Validation
- 🔎 Exploratory Data Analysis
- 📈 Business Intelligence
- 🤖 Machine Learning
- 🎯 Churn Prediction
- 📉 Watch Duration Prediction
- ⭐ Rating Prediction
- 🎬 Movie Recommendation
- 📊 Advanced Visualization
- 📋 Business KPI Analysis

The project works with six interconnected datasets representing subscribers, content, viewing behavior, ratings, reviews and watchlists.

---

# 🎯 Project Objective

The main objective of StreamFlix is to transform raw streaming-platform data into:

```text
RAW DATA
    ↓
DATA VALIDATION
    ↓
DATA CLEANING
    ↓
EXPLORATORY DATA ANALYSIS
    ↓
BUSINESS KPIs
    ↓
ADVANCED ANALYTICS
    ↓
MACHINE LEARNING
    ↓
RECOMMENDATION SYSTEM
    ↓
BUSINESS INSIGHTS
```

The project focuses on understanding:

- Who is using the platform?
- What content are users watching?
- How long are they watching?
- Which genres and titles perform well?
- Which plans generate more revenue?
- What devices are being used?
- What factors relate to subscriber activity?
- Can watch duration be predicted?
- Can ratings be predicted?
- Can movies be recommended based on genre similarity?

---

# 🗂️ Dataset Architecture

StreamFlix uses **six interconnected CSV tables**:

```text
                         STREAMFLIX
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
     Subscribers           Titles          Watch History
          │                   │                   │
          └──────────────┬────┴───────┬──────────┘
                         │            │
                         ▼            ▼
                      Ratings       Reviews
                         │
                         ▼
                      Watchlist
```

---

# 📁 Six Core Datasets

## 👥 1. Subscribers

Contains subscriber-level information such as:

- Subscriber ID
- Signup date
- Country
- Region
- Age
- Gender
- Plan type
- Monthly price
- Household size
- Primary device
- Payment method
- Tenure
- Active status
- Churn date

---

## 🎬 2. Titles

Contains content-level information such as:

- Title ID
- Title name
- Content type
- Genre
- Country
- Language
- Release year
- Date added
- Maturity rating
- Seasons
- Content duration
- Original content
- License type
- Director
- Cast
- Quality score
- Popularity score
- License cost
- Total watch hours
- Total plays

---

## 👀 3. Watch History

Contains user viewing activity:

- Watch ID
- Subscriber ID
- Title ID
- Watch date
- Device
- Region
- Content duration
- Watch duration
- Completion percentage
- Completed status

---

## ⭐ 4. Ratings

Contains user ratings:

- Rating ID
- Subscriber ID
- Title ID
- Rating
- Rating date

---

## 💬 5. Reviews

Contains user reviews and sentiment information:

- Review ID
- Subscriber ID
- Title ID
- Review text
- Sentiment
- Helpful votes
- Review date

---

## 📺 6. Watchlist

Contains user watchlist activity:

- Watchlist ID
- Subscriber ID
- Title ID
- Added date
- Watched status

---

# 🔐 Data Validation

The project validates the expected schema of all six datasets before analysis.

It checks whether required columns exist in:

```text
Subscribers
Titles
Watch History
Ratings
Reviews
Watchlist
```

The project also performs **referential integrity checks** between subscribers and titles and their related event tables.

```text
Watch History
      ↓
Subscriber ID ─────→ Subscribers
      ↓
Title ID ───────────→ Titles

Ratings
      ↓
Subscriber ID ─────→ Subscribers
Title ID ──────────→ Titles
```

---

# 📊 Exploratory Data Analysis

The project performs detailed analysis on every major table.

### Dataset Analysis

For each dataset, the project examines:

- Rows
- Columns
- Data types
- Missing values
- Duplicate records
- Unique values
- Statistical summaries
- Correlations

---

# 📈 Business KPIs

StreamFlix calculates important business metrics including:

### 👥 Subscriber KPIs

- Total Subscribers
- Active Subscribers
- Inactive Subscribers
- Active Rate
- Churn Rate

### 🎬 Content KPIs

- Total Titles
- Movies
- TV Shows
- Original Content
- Licensed Content

### 👀 Viewing KPIs

- Total Watch Sessions
- Total Watch Hours
- Average Watch Duration
- Average Completion
- Completed Views
- Completion Rate

### ⭐ Engagement KPIs

- Total Ratings
- Average Rating
- Highest Rating
- Lowest Rating
- Total Reviews

### 📺 Watchlist KPIs

- Total Watchlist Entries
- Watched Entries
- Unwatched Entries
- Watchlist Conversion Rate

### 💰 Revenue Analysis

- Monthly subscription revenue
- Revenue by plan type

---

# 📊 Advanced Data Visualization

The project includes multiple analytical visualizations.

## 🎭 Genre Analysis

```text
Top 10 Genres
       ↓
Watch Duration
       ↓
Genre Performance
```

Analyzes which genres generate the most watch time.

---

## 🌍 Country Analysis

Analyzes watch time across countries and identifies the top countries by viewing activity.

---

## 💳 Plan Analysis

Analyzes watch hours and revenue according to subscription plans.

---

## 📱 Device Analysis

Analyzes which devices are most frequently used for streaming.

Example categories may include:

```text
Mobile
TV
Laptop
Tablet
```

---

## 🌐 Language Analysis

Analyzes the most common content languages available in the platform data.

---

# 📅 Time-Series Analysis

The project analyzes viewing behavior over time.

### Monthly Viewing Trend

Tracks the number of viewing sessions by month.

### Monthly Watch Hours

Tracks total watch hours by month.

```text
Month
  ↓
Viewing Sessions
  ↓
Watch Hours
  ↓
Trend Analysis
```

---

# 🎬 Content Performance

The project identifies:

- Top 10 most-watched titles
- Top genres
- Content type distribution
- Language distribution
- Original vs licensed content
- Content popularity

---

# 🔥 Correlation Analysis

Correlation analysis is performed on numerical variables to identify relationships within the data.

Example:

```text
Watch Duration
      ↕
Completion %
```

The project also generates correlation heatmaps for numerical features.

---

# 🚨 Outlier Detection

Box plots are used to identify potential outliers in:

```text
Watch Duration
```

This helps identify unusually short or long viewing sessions.

---

# 📈 Watch Duration vs Completion

A scatter plot is used to analyze the relationship between:

```text
Watch Duration
       ↕
Completion %
```

This helps explore user viewing behavior.

---

# 🤖 Machine Learning

StreamFlix contains multiple Machine Learning components.

---

# 1️⃣ Churn / Subscriber Activity Prediction

### 🎯 Objective

Build a classification model using subscriber information to predict the subscriber activity target represented by `is_active`.

### Model

```text
Random Forest Classifier
```

### Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

### Features

The model uses:

```text
Age
Monthly Price
Household Size
Tenure Months
Country
Region
Gender
Plan Type
Primary Device
Payment Method
```

### Evaluation

The project calculates:

- Accuracy
- Confusion Matrix
- Classification Report
- Feature Importance

---

# 2️⃣ Watch Duration Prediction

### 🎯 Objective

Predict the expected watch duration of a viewing session.

### Model

```text
Linear Regression
```

### Target

```text
watch_duration_min
```

### Evaluation Metrics

```text
R² Score
MAE
RMSE
```

### Visualizations

- Actual vs Predicted
- Feature Coefficients
- Residual Error Plot

---

# 3️⃣ Rating Prediction

### 🎯 Objective

Predict user ratings using viewing-history information.

The project combines:

```text
Ratings
      +
Watch History
```

using:

```text
subscriber_id
title_id
```

### Model

```text
Random Forest Regressor
```

### Evaluation

```text
R² Score
MAE
RMSE
```

### Additional Analysis

- Feature importance
- Actual vs predicted ratings
- Prediction samples

---

# 🎬 4️⃣ Movie Recommendation System

StreamFlix also contains a content-based recommendation system.

### Technology

```text
CountVectorizer
       ↓
Genre Matrix
       ↓
Cosine Similarity
       ↓
Movie Recommendations
```

The recommendation system uses the movie's:

```text
Primary Genre
```

to calculate similarity between titles.

### Example Workflow

```text
User selects a movie
        ↓
Find movie
        ↓
Compare genre representation
        ↓
Calculate cosine similarity
        ↓
Rank similar movies
        ↓
Return recommendations
```

The system returns the top recommended movies based on similarity.

---

# 🧠 AI / Machine Learning Architecture

```text
                    STREAMFLIX DATA
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        Subscribers    Watch History   Ratings
             │             │             │
             ▼             ▼             ▼
        Classification  Regression    Regression
             │             │             │
             ▼             ▼             ▼
        Activity       Watch Time     Rating
        Prediction     Prediction     Prediction
                           │
                           ▼
                  Recommendation System
                           │
                           ▼
                    Movie Suggestions
```

---

# 📊 Professional Dashboards

The project includes three dashboard concepts:

## 🏢 Dashboard 1 — Executive / Business Overview

Focuses on:

- Subscribers
- Watch hours
- Revenue
- Content
- Ratings
- Business KPIs

---

## 🎬 Dashboard 2 — Content & User Intelligence

Focuses on:

- Content performance
- Genres
- Countries
- Languages
- Devices
- User engagement
- Watch behavior

---

## 🤖 Dashboard 3 — AI / Machine Learning Lab

Focuses on:

- Churn/activity prediction
- Watch duration prediction
- Rating prediction
- Feature importance
- Model evaluation
- Recommendation system

---

# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualization |
| Streamlit | Dashboard development |
| Scikit-learn | Machine Learning |
| CountVectorizer | Text/genre representation |
| Cosine Similarity | Recommendation |
| CSV | Data storage |

---

# 📁 Recommended Repository Structure

```text
StreamFlix-Data-Science/
│
├── data/
│   ├── ratings.csv
│   ├── reviews.csv
│   ├── subscribers.csv
│   ├── titles.csv
│   ├── watch_history.csv
│   └── watchlist.csv
│
├── src/
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── machine_learning.py
│   └── recommendation.py
│
├── dashboards/
│   ├── executive_dashboard.py
│   ├── content_dashboard.py
│   └── ml_dashboard.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_business_analysis.ipynb
│   ├── 04_machine_learning.ipynb
│   └── 05_recommendation_system.ipynb
│
├── models/
│   ├── churn_model/
│   ├── watch_duration_model/
│   └── rating_model/
│
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── predictions/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/StreamFlix-Data-Science.git
```

Move into the project:

```bash
cd StreamFlix-Data-Science
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
pip install pandas numpy matplotlib seaborn streamlit scikit-learn plotly
```

Or:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

If the main application is a Streamlit dashboard:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

Make sure the six CSV files are available in the configured project data folder.

---

# 📦 Required Dataset Files

```text
ratings.csv
reviews.csv
subscribers.csv
titles.csv
watch_history.csv
watchlist.csv
```

The project loader can also recognize downloaded variants such as:

```text
ratings(2).csv
subscribers(2).csv
titles(2).csv
```

and searches several project/data locations instead of depending on a hard-coded Windows username path.

---

# 🔍 Data Quality Checks

The project performs:

```text
Schema Validation
       ↓
Missing Value Analysis
       ↓
Duplicate Analysis
       ↓
Unique Value Analysis
       ↓
Statistical Analysis
       ↓
Referential Integrity
```

This helps ensure that relationships between the six datasets remain valid.

---

# 💡 Business Questions Answered

The project is designed to answer questions such as:

### 👥 Subscribers

- How many subscribers are active?
- What subscription plans are being used?
- Which countries have the most subscribers?
- Which devices are commonly used?
- What is the subscriber churn/activity rate?

### 🎬 Content

- Which genres receive the most watch time?
- Which titles are watched the most?
- What languages are most represented?
- How many movies and TV shows exist?
- How much content is original vs licensed?

### 👀 Engagement

- How many total viewing sessions occur?
- What is the average watch duration?
- What is the average completion percentage?
- Which devices generate the most viewing activity?
- How does viewing change over time?

### ⭐ Ratings & Reviews

- What is the average rating?
- What is the rating distribution?
- How many reviews are available?
- What is the sentiment distribution?
- Which reviews receive helpful votes?

### 💰 Business

- Which plans generate more revenue?
- What is the watchlist conversion rate?
- How does subscriber activity relate to platform usage?

---

# 📊 Project Workflow

```text
                STREAMFLIX
                    │
                    ▼
              DATA INGESTION
                    │
                    ▼
             DATA VALIDATION
                    │
                    ▼
              DATA ANALYSIS
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        EDA               BUSINESS KPI
          │                   │
          └─────────┬─────────┘
                    ▼
             DATA VISUALIZATION
                    │
                    ▼
             FEATURE ENGINEERING
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       CHURN      WATCH      RATING
    PREDICTION   DURATION   PREDICTION
          │         │         │
          └─────────┼─────────┘
                    ▼
          RECOMMENDATION SYSTEM
                    │
                    ▼
             DASHBOARD / REPORT
```

---

# 🚀 Future Enhancements

Possible future versions can include:

### 🤖 AI

- Advanced NLP review analysis
- Sentiment classification
- Deep Learning recommendation models
- Personalized recommendations
- User-level content ranking

### 📊 Analytics

- Real-time dashboard
- Interactive filters
- Geographic maps
- Cohort analysis
- Retention analysis
- Customer lifetime value

### 🧠 Machine Learning

- XGBoost
- Gradient Boosting
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Model deployment

### 🌐 Deployment

- Streamlit Cloud
- Docker
- REST API
- Cloud database
- Automated model pipeline

---

# 📚 Skills Demonstrated

This project demonstrates practical experience with:

```text
Python
Pandas
NumPy
Data Cleaning
Data Validation
EDA
Statistics
Data Visualization
Business Analytics
Machine Learning
Classification
Regression
Recommendation Systems
Streamlit
Dashboard Development
```

---

# 🎓 Learning Outcomes

Through this project, the complete Data Science lifecycle is demonstrated:

```text
Understand
    ↓
Collect
    ↓
Validate
    ↓
Clean
    ↓
Analyze
    ↓
Visualize
    ↓
Model
    ↓
Evaluate
    ↓
Recommend
    ↓
Communicate
```

---

# ⚠️ Important Note

This is an analytics and machine-learning project built for educational and portfolio purposes.

Machine Learning outputs are predictions based on the available dataset and should be interpreted as analytical results rather than guaranteed future outcomes.

---

# 👨‍💻 Author

## Abdullah Ghufran

**Data Science | Data Analytics | Python | SQL | Machine Learning | Power BI**

---

# ⭐ Project Support

If you find this project useful:

⭐ Star the repository  
🍴 Fork the repository  
🐛 Report issues  
💡 Suggest improvements  

---

# 🎬 StreamFlix

### From Streaming Data to Business Intelligence

```text
DATA
 ↓
INSIGHTS
 ↓
MACHINE LEARNING
 ↓
RECOMMENDATIONS
 ↓
DECISIONS
```

> **Analyze the audience. Understand the content. Predict the behavior. Build smarter recommendations.**
