# 🎬 STREAMFLIX
## Data Science, Machine Learning & Business Intelligence Platform

A comprehensive end-to-end Data Science project designed to analyze the complete ecosystem of a movie streaming platform.

**STREAMFLIX** combines **Data Analysis, Exploratory Data Analysis (EDA), Business Intelligence, Machine Learning, Predictive Analytics, Recommendation Systems, and Professional Dashboard Development** into one integrated analytics platform.

The project transforms raw streaming platform data into meaningful business insights, predictive models, intelligent recommendations, and interactive visual analytics.

---

# 🚀 Project Overview

The STREAMFLIX platform analyzes multiple aspects of a streaming business, including:

- 👥 Subscriber behavior
- 🎬 Movie and content performance
- ⭐ Ratings and audience engagement
- 💬 User reviews and feedback
- 📺 Watch history and viewing patterns
- 📌 Watchlist activity
- 📈 Business KPIs
- 🤖 Machine Learning predictions
- 🎯 Movie recommendations

The project follows a complete Data Science workflow:

```text
Raw Data
   ↓
Data Loading
   ↓
Data Quality Checks
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Business KPI Analysis
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Recommendation System
   ↓
Professional Dashboards
   ↓
Business Insights# 🎬 STREAMFLIX
## Data Science, Machine Learning & Business Intelligence Platform

A comprehensive end-to-end Data Science project designed to analyze the complete ecosystem of a movie streaming platform.

**STREAMFLIX** combines **Data Analysis, Exploratory Data Analysis (EDA), Business Intelligence, Machine Learning, Predictive Analytics, Recommendation Systems, and Professional Dashboard Development** into one integrated analytics platform.

The project transforms raw streaming platform data into meaningful business insights, predictive models, intelligent recommendations, and interactive visual analytics.

---

# 🚀 Project Overview

The STREAMFLIX platform analyzes multiple aspects of a streaming business, including:

- 👥 Subscriber behavior
- 🎬 Movie and content performance
- ⭐ Ratings and audience engagement
- 💬 User reviews and feedback
- 📺 Watch history and viewing patterns
- 📌 Watchlist activity
- 📈 Business KPIs
- 🤖 Machine Learning predictions
- 🎯 Movie recommendations

The project follows a complete Data Science workflow:

```text
Raw Data
   ↓
Data Loading
   ↓
Data Quality Checks
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Business KPI Analysis
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Recommendation System
   ↓
Professional Dashboards
   ↓
Business Insights# 🎬 STREAMFLIX
## Data Science, Machine Learning & Business Intelligence Platform

A comprehensive end-to-end Data Science project designed to analyze the complete ecosystem of a movie streaming platform.

**STREAMFLIX** combines **Data Analysis, Exploratory Data Analysis (EDA), Business Intelligence, Machine Learning, Predictive Analytics, Recommendation Systems, and Professional Dashboard Development** into one integrated analytics platform.

The project transforms raw streaming platform data into meaningful business insights, predictive models, intelligent recommendations, and interactive visual analytics.

---

# 🚀 Project Overview

The STREAMFLIX platform analyzes multiple aspects of a streaming business, including:

- 👥 Subscriber behavior
- 🎬 Movie and content performance
- ⭐ Ratings and audience engagement
- 💬 User reviews and feedback
- 📺 Watch history and viewing patterns
- 📌 Watchlist activity
- 📈 Business KPIs
- 🤖 Machine Learning predictions
- 🎯 Movie recommendations

The project follows a complete Data Science workflow:

```text
Raw Data
   ↓
Data Loading
   ↓
Data Quality Checks
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Business KPI Analysis
   ↓
Feature Engineering
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Recommendation System
   ↓
Professional Dashboards
   ↓
Business Insights
📂 Dataset Structure

The project works with six interconnected datasets:

STREAMFLIX/
│
├── ratings.csv
├── reviews.csv
├── subscribers.csv
├── titles.csv
├── watch_history.csv
└── watchlist.csv
Dataset Description
Dataset	Description
subscribers.csv	Subscriber demographics, plans, account and activity information
titles.csv	Movie and streaming content information
ratings.csv	User ratings for movies and titles
reviews.csv	Audience reviews and feedback
watch_history.csv	User viewing activity and watch duration
watchlist.csv	Movies saved by users for future viewing
🛠️ Technologies Used
🐍 Programming Language
Python
📊 Data Analysis & Processing
Pandas
NumPy
📈 Data Visualization
Matplotlib
Seaborn
🤖 Machine Learning
Scikit-learn
🎯 Recommendation System
CountVectorizer
Cosine Similarity
🖥️ Dashboard Development
Streamlit
⚙️ Key Project Features
1️⃣ Intelligent Data Loading System

The project includes a robust and flexible data loading system that automatically searches for the required CSV files.

Instead of relying only on hard-coded paths, the system checks multiple possible locations:

1. Custom environment directory
2. Project folder
3. Current working directory
4. Downloads project folder
5. Downloads directory
6. Desktop directory

It can also handle renamed or downloaded dataset variants, making the project more portable and easier to run on different systems.

🧹 2️⃣ Data Quality & Cleaning

Before performing analysis, the project performs important data quality checks.

Quality Checks Include:
Missing value detection
Duplicate row detection
Dataset profiling
Row and column analysis
Data type validation
Date conversion
Data cleaning and preprocessing

The project prepares structured datasets for analysis and machine learning.

📊 3️⃣ Exploratory Data Analysis

The EDA section helps understand patterns, relationships, trends, and user behavior across the STREAMFLIX platform.

The analysis focuses on:

Subscriber distribution
Content categories
Ratings
Watch duration
Completion percentage
Viewing behavior
Device usage
Audience engagement
Correlation between numerical features
💼 4️⃣ Business Intelligence & KPI Analysis

STREAMFLIX calculates important business metrics to provide an executive-level overview of the platform.

Key Performance Indicators
👥 Total Subscribers
🎬 Total Titles
⭐ Total Ratings
💬 Total Reviews
📺 Total Watch Sessions
📌 Total Watchlist Activity
⭐ Average Rating
🎯 Average Completion Percentage
⏱️ Total Watch Hours

These KPIs provide a high-level understanding of platform performance, audience engagement, and content consumption.

🤖 5️⃣ Machine Learning Models

The project includes multiple Machine Learning implementations for prediction and intelligent analysis.

🔵 A. Subscriber Activity / Churn Classification

A Random Forest Classifier is used to predict subscriber activity and potential churn behavior.

Features Used
Age
Monthly Price
Household Size
Tenure
Country
Region
Gender
Plan Type
Primary Device
Payment Method
Machine Learning Workflow
Data Cleaning
      ↓
Categorical Encoding
      ↓
Boolean Conversion
      ↓
Feature Selection
      ↓
Train-Test Split
      ↓
Random Forest Training
      ↓
Prediction
      ↓
Model Evaluation
Evaluation Metrics
Accuracy Score
Confusion Matrix
Classification Report
Feature Importance
🟢 B. Watch Duration Prediction

A Linear Regression Model is used to predict user watch duration.

The model analyzes relationships between available viewing features and watch duration.

Evaluation Metrics
R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Visual Analysis
Actual vs Predicted Values
Feature Coefficient Analysis
Residual Error Analysis
🟣 C. Rating Prediction

A Random Forest Regressor is used to predict movie ratings.

The project combines rating and watch history information to create a richer dataset for predictive analysis.

Model Analysis
Actual vs Predicted Ratings
Feature Importance
Prediction Performance Analysis
🎯 6️⃣ Movie Recommendation System

STREAMFLIX includes a content-based movie recommendation system.

The recommendation engine uses movie genre information to identify similar content.

Movie Genre
      ↓
CountVectorizer
      ↓
Genre Matrix
      ↓
Cosine Similarity
      ↓
Similarity Ranking
      ↓
Top Movie Recommendations

The user can enter a movie name, and the system returns similar movies based on content characteristics.

Recommendation Workflow
User Selects Movie
        ↓
Movie Genre Extracted
        ↓
CountVectorizer Applied
        ↓
Cosine Similarity Calculated
        ↓
Similar Movies Ranked
        ↓
Top Recommendations Displayed
🖥️ 7️⃣ Professional Dashboard System

The project includes multiple professional dashboard interfaces for different levels of analysis.

📊 Dashboard 1 — Executive Business Overview

Designed for high-level business analysis and management decision-making.

Focus Areas
Overall business performance
Subscriber metrics
Content performance
Platform KPIs
Business trends
Audience engagement
👥 Dashboard 2 — Content & User Intelligence

Designed to understand audience behavior and content performance.

Focus Areas
User engagement
Content performance
Viewing behavior
Device usage
Ratings
Reviews
Completion behavior
🤖 Dashboard 3 — AI & Machine Learning Lab

Designed to visualize Machine Learning model performance.

Includes
Model performance metrics
Confusion Matrix
Feature Importance
Correlation Analysis
Prediction Error Distribution
Actual vs Predicted Analysis
Model Residual Analysis
🎬 8️⃣ Selected Movie Analytics Dashboard

One of the major features of STREAMFLIX is the ability to analyze an individual movie.

The system accepts a movie name and automatically filters the related data.

Movie-Level KPIs
👁️ Total Views
⏱️ Total Watch Hours
⭐ Average Rating
🎯 Average Completion
💬 Total Reviews
Movie-Level Analysis
📈 Viewing Trend

Displays daily viewing activity for the selected movie.

📱 Device Usage

Shows which devices are used by viewers.

🎯 Completion Analysis

Compares:

Completed Views
        vs
Incomplete Views
⭐ Rating Distribution

Analyzes audience rating patterns.

💬 Review Analysis

Displays audience feedback information when review data is available.

📐 Project Architecture
                         ┌───────────────────┐
                         │   RAW DATASETS    │
                         │  6 CSV FILES      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   DATA LOADER     │
                         │ Intelligent Path  │
                         │    Detection      │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ DATA CLEANING &   │
                         │ QUALITY CHECKS    │
                         └─────────┬─────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  ▼                ▼                ▼
           ┌────────────┐    ┌────────────┐   ┌────────────┐
           │    EDA     │    │ BUSINESS   │   │ MACHINE    │
           │ ANALYSIS   │    │    KPI     │   │ LEARNING   │
           └─────┬──────┘    └─────┬──────┘   └─────┬──────┘
                 │                 │                │
                 └─────────────────┼────────────────┘
                                   ▼
                         ┌───────────────────┐
                         │ RECOMMENDATION    │
                         │     SYSTEM        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ PROFESSIONAL      │
                         │   DASHBOARDS      │
                         └───────────────────┘
📈 Data Visualizations

The project generates multiple types of visualizations, including:

Bar Charts
Line Charts
Pie Charts
Donut Charts
Scatter Plots
Histograms
Correlation Matrices
Confusion Matrices
Feature Importance Graphs
Actual vs Predicted Plots
Residual Error Plots
📏 Model Evaluation
Classification Model
Accuracy Score
Confusion Matrix
Classification Report
Feature Importance
Regression Models
R² Score
Mean Absolute Error
Mean Squared Error
Root Mean Squared Error
Actual vs Predicted Analysis
Residual Analysis
📦 Installation

Clone the repository:

git clone https://github.com/abdullahghufran2001-hub/streamflix-data-science-project.git

Move into the project directory:

cd streamflix-data-science-project

Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn streamlit
▶️ How to Run

Make sure the following files are available in the project directory:

ratings.csv
reviews.csv
subscribers.csv
titles.csv
watch_history.csv
watchlist.csv

Run the project:

python streamflix.py

The application will:

Load the datasets
Perform data quality checks
Clean and preprocess the data
Perform exploratory data analysis
Generate business KPIs
Train Machine Learning models
Evaluate model performance
Run the recommendation system
Generate dashboards
Produce movie-level analytics
📁 Recommended Repository Structure
streamflix-data-science-project/
│
├── streamflix.py
│
├── ratings.csv
├── reviews.csv
├── subscribers.csv
├── titles.csv
├── watch_history.csv
├── watchlist.csv
│
├── screenshots/
│   ├── executive_dashboard.png
│   ├── content_user_dashboard.png
│   ├── ai_ml_dashboard.png
│   └── movie_analytics.png
│
├── README.md
│
└── requirements.txt
📦 requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
🧠 Skills Demonstrated

This project demonstrates practical knowledge of:

Python Programming
Data Loading
Data Cleaning
Data Quality Assessment
Exploratory Data Analysis
Data Visualization
Business Intelligence
KPI Development
Feature Engineering
Categorical Encoding
Machine Learning Classification
Machine Learning Regression
Random Forest
Linear Regression
Model Evaluation
Feature Importance Analysis
Residual Analysis
Recommendation Systems
Cosine Similarity
Professional Dashboard Design
🎯 Key Learning Outcomes

This project demonstrates the complete Data Science journey:

Data → Analysis → Insights → Prediction → Intelligence

Instead of focusing on only one dataset or one Machine Learning model, STREAMFLIX integrates multiple Data Science concepts into one real-world business environment.

The project combines:

Data Analysis
Exploratory Data Analysis
Business Intelligence
Machine Learning
Predictive Analytics
Recommendation Systems
Visual Analytics
🔮 Future Improvements

Future versions of STREAMFLIX may include:

Deep Learning Recommendation Models
Natural Language Processing for Review Analysis
Advanced Sentiment Analysis
Real-Time Streaming Analytics
SQL Database Integration
Cloud Deployment
Advanced Streamlit Web Application
Time Series Forecasting
Advanced Customer Churn Prediction
MLOps Pipeline
👨‍💻 Author
Abdullah Ghufran

Aspiring Data Scientist | Data Analyst | Python Developer | Machine Learning Enthusiast

🌐 GitHub: https://github.com/abdullahghufran2001-hub

⭐ Support

If you found this project useful or interesting:

⭐ Star the repository
🍴 Fork the project
📢 Share it with others

📌 Conclusion

STREAMFLIX is an end-to-end Data Science project that demonstrates the complete journey from raw streaming platform data to meaningful business insights and intelligent predictions.

The project combines:

Data Analysis + Machine Learning + Recommendation System + Business Intelligence + Professional Dashboard Development

to create a comprehensive analytics solution for a modern streaming platform.

