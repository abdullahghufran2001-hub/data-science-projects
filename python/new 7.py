
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor




from sklearn.ensemble import RandomForestClassifier
# =====================================================
# MACHINE LEARNING - LINEAR REGRESSION
# WATCH DURATION PREDICTION
# =====================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# MODEL EVALUATION METRICS
# =====================================================

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# DATA FILE LOADER
# =====================================================
# Put the six CSV files in the same folder as this Python file,
# or change DATA_DIR below to the folder containing your CSVs.
from pathlib import Path
import os

# =====================================================
# STREAMFLIX — ROBUST DATA PATH / FILE LOADER
# =====================================================
# FIX:
# The old code contained a hard-coded path:
# C:/Users/user/Downloads/project file
# This fails when the Windows username or project location is different.
#
# The new loader automatically checks:
# 1. STREAMFLIX_DATA_DIR environment variable (if set)
# 2. Folder containing this Python file
# 3. Current working directory
# 4. Downloads/project file
# 5. Downloads
# 6. Desktop
#
# It also accepts renamed/downloaded files such as:
# ratings(4).csv, subscribers(2).csv, titles(2).csv, etc.
# =====================================================

DATA_DIR = Path(__file__).resolve().parent

_REQUIRED_FILES = [
    "ratings.csv",
    "reviews.csv",
    "subscribers.csv",
    "titles.csv",
    "watch_history.csv",
    "watchlist.csv",
]

def _candidate_data_dirs():
    """Return likely StreamFlix data folders without hard-coded usernames."""
    candidates = []

    # User can explicitly control the data folder.
    env_dir = os.environ.get("STREAMFLIX_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    # Folder containing this .py file.
    candidates.append(DATA_DIR)

    # Folder from which Python was launched.
    candidates.append(Path.cwd())

    # Standard Windows locations for this project.
    home = Path.home()
    candidates.extend([
        home / "Downloads" / "project file",
        home / "Downloads" / "Project file",
        home / "Downloads",
        home / "Desktop",
    ])

    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for folder in candidates:
        try:
            folder = folder.resolve()
        except Exception:
            folder = Path(folder)
        key = str(folder).lower()
        if key not in seen:
            seen.add(key)
            unique.append(folder)

    return unique


def _find_project_csv(filename):
    """Find an exact CSV first, then a safe renamed variant."""
    stem = Path(filename).stem

    for folder in _candidate_data_dirs():
        # 1) Exact expected filename.
        exact = folder / filename
        if exact.is_file():
            return exact

        # 2) Downloaded variants, e.g. subscribers(2).csv.
        try:
            variants = sorted(
                p for p in folder.glob(stem + "*.csv")
                if p.is_file()
            )
        except Exception:
            variants = []

        if variants:
            return variants[0]

    return None


def load_project_csv(filename):
    """Load one StreamFlix CSV with a clear error if it cannot be found."""
    path = _find_project_csv(filename)

    if path is None:
        searched = "\n".join(f"  - {p}" for p in _candidate_data_dirs())
        raise FileNotFoundError(
            f"\nCould not find StreamFlix file: {filename}\n\n"
            f"Folders searched:\n{searched}\n\n"
            "FIX: Keep all six CSV files beside this Python file, "
            "or set STREAMFLIX_DATA_DIR to the folder containing them."
        )

    print(f"Loaded {filename}  <--  {path}")
    return pd.read_csv(path)


# =====================================================
# LOAD ALL SIX STREAMFLIX TABLES
# =====================================================
# ratings is kept as both `df` and `ratings` so the original
# analysis code continues to work while matching the database schema.
df = load_project_csv("ratings.csv")
ratings = df

reviews = load_project_csv("reviews.csv")
subscribers = load_project_csv("subscribers.csv")
titles = load_project_csv("titles.csv")
watch_history = load_project_csv("watch_history.csv")
watchlist = load_project_csv("watchlist.csv")


# =====================================================
# SCHEMA / COLUMN VALIDATION
# =====================================================
# These are the columns defined in the StreamFlix Data Dictionary.
_EXPECTED_COLUMNS = {
    "subscribers": {
        "subscriber_id", "signup_date", "country", "region", "age",
        "gender", "plan_type", "monthly_price_usd", "household_size",
        "primary_device", "payment_method", "tenure_months",
        "is_active", "churn_date"
    },
    "titles": {
        "title_id", "title_name", "type", "primary_genre", "country",
        "language", "release_year", "date_added", "maturity_rating",
        "seasons", "content_duration_min", "is_original", "license_type",
        "director", "cast", "quality_score", "popularity_score",
        "license_cost_usd", "license_expiry", "total_watch_hours",
        "total_plays"
    },
    "watch_history": {
        "watch_id", "subscriber_id", "title_id", "watch_date", "device",
        "region", "content_duration_min", "watch_duration_min",
        "completion_pct", "completed"
    },
    "ratings": {
        "rating_id", "subscriber_id", "title_id", "rating", "rating_date"
    },
    "reviews": {
        "review_id", "subscriber_id", "title_id", "review_text",
        "sentiment", "helpful_votes", "review_date"
    },
    "watchlist": {
        "watchlist_id", "subscriber_id", "title_id", "added_date", "watched"
    },
}

_loaded_tables = {
    "subscribers": subscribers,
    "titles": titles,
    "watch_history": watch_history,
    "ratings": ratings,
    "reviews": reviews,
    "watchlist": watchlist,
}

_schema_errors = []
for _table_name, _expected in _EXPECTED_COLUMNS.items():
    _actual = set(_loaded_tables[_table_name].columns)
    _missing = sorted(_expected - _actual)
    if _missing:
        _schema_errors.append(
            f"{_table_name}: missing columns -> {_missing}"
        )

if _schema_errors:
    raise ValueError(
        "\nStreamFlix schema validation failed.\n" +
        "\n".join(_schema_errors)
    )

print("\n" + "=" * 70)
print("STREAMFLIX — ALL 6 TABLES LOADED SUCCESSFULLY")
print("=" * 70)
for _name, _table in _loaded_tables.items():
    print(f"{_name:15s}: {_table.shape[0]:,} rows × {_table.shape[1]} columns")
print("=" * 70)


# =====================================================
# REFERENTIAL-INTEGRITY CHECK
# =====================================================
# Every event table must point to a real subscriber and title.
def _count_orphans(child_df, child_key, parent_df, parent_key):
    return int((~child_df[child_key].isin(parent_df[parent_key])).sum())


_fk_checks = {
    "watch_history -> subscribers": _count_orphans(
        watch_history, "subscriber_id", subscribers, "subscriber_id"
    ),
    "watch_history -> titles": _count_orphans(
        watch_history, "title_id", titles, "title_id"
    ),
    "ratings -> subscribers": _count_orphans(
        ratings, "subscriber_id", subscribers, "subscriber_id"
    ),
    "ratings -> titles": _count_orphans(
        ratings, "title_id", titles, "title_id"
    ),
    "reviews -> subscribers": _count_orphans(
        reviews, "subscriber_id", subscribers, "subscriber_id"
    ),
    "reviews -> titles": _count_orphans(
        reviews, "title_id", titles, "title_id"
    ),
    "watchlist -> subscribers": _count_orphans(
        watchlist, "subscriber_id", subscribers, "subscriber_id"
    ),
    "watchlist -> titles": _count_orphans(
        watchlist, "title_id", titles, "title_id"
    ),
}

print("\n===== REFERENTIAL INTEGRITY CHECK =====")
for _relationship, _orphans in _fk_checks.items():
    status = "PASS" if _orphans == 0 else "CHECK"
    print(f"{status:5s} | {_relationship}: {_orphans:,} orphan rows")


# Display Data
print("===== COMPLETE DATA =====")
print(df)


print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== LAST 5 ROWS =====")
print(df.tail())

print("\n===== RANDOM 5 ROWS =====")
print(df.sample(5))

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== SHAPE (ROWS, COLUMNS) =====")
print(df.shape)

print("\n===== NUMBER OF ROWS =====")
print(df.shape[0])

print("\n===== NUMBER OF COLUMNS =====")
print(df.shape[1])

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== TOTAL MISSING VALUES =====")
print(df.isnull().sum().sum())

print("\n===== DUPLICATE VALUES =====")
print(df.duplicated().sum())

print("\n===== UNIQUE VALUES =====")
print(df.nunique())

print("\n===== STATISTICAL SUMMARY =====")
print(pd.DataFrame({
    "Count": df.count(),
    "Sum": df.sum(numeric_only=True),
    "Mean": df.mean(numeric_only=True),
    "Median": df.median(numeric_only=True),
    "Min": df.min(numeric_only=True),
    "Max": df.max(numeric_only=True),
    "Std": df.std(numeric_only=True),
    "Variance": df.var(numeric_only=True)
}))

print("\n===== DESCRIBE =====")
print(df.describe(include="all"))

print("\n===== CORRELATION =====")
print(df.corr(numeric_only=True))
# ==========================================
# RATINGS SUMMARY
# ==========================================

print("\n===== TOTAL RATINGS =====")
print(len(df))

print("\n===== AVERAGE RATING =====")
print(df["rating"].mean())

print("\n===== MAX RATING =====")
print(df["rating"].max())

print("\n===== MIN RATING =====")
print(df["rating"].min())

print("\n===== RATING DISTRIBUTION =====")
print(df["rating"].value_counts().sort_index())

#=============reviews================#

print("===== COMPLETE DATA =====")
print(reviews)

print("\n===== FIRST 5 ROWS =====")
print(reviews.head())

print("\n===== LAST 5 ROWS =====")
print(reviews.tail())

print("\n===== RANDOM 5 ROWS =====")
print(reviews.sample(5))

print("\n===== COLUMN NAMES =====")
print(reviews.columns)

print("\n===== DATA TYPES =====")
print(reviews.dtypes)

print("\n===== SHAPE (ROWS, COLUMNS) =====")
print(reviews.shape)

print("\n===== NUMBER OF ROWS =====")
print(reviews.shape[0])

print("\n===== NUMBER OF COLUMNS =====")
print(reviews.shape[1])

print("\n===== MISSING VALUES =====")
print(reviews.isnull().sum())

print("\n===== TOTAL MISSING VALUES =====")
print(reviews.isnull().sum().sum())

print("\n===== DUPLICATE VALUES =====")
print(reviews.duplicated().sum())

print("\n===== UNIQUE VALUES =====")
print(reviews.nunique())

print("\n===== STATISTICAL SUMMARY =====")
print(pd.DataFrame({
    "Count": reviews.count(),
    "Sum": reviews.sum(numeric_only=True),
    "Mean": reviews.mean(numeric_only=True),
    "Median": reviews.median(numeric_only=True),
    "Min": reviews.min(numeric_only=True),
    "Max": reviews.max(numeric_only=True),
    "Std": reviews.std(numeric_only=True),
    "Variance": reviews.var(numeric_only=True)
}))

print("\n===== DESCRIBE =====")
print(reviews.describe(include="all"))

print("\n===== CORRELATION =====")
print(reviews.corr(numeric_only=True))
# ==========================================
# REVIEW SENTIMENT SUMMARY
# ==========================================

print("\n===== TOTAL REVIEWS =====")
print(len(reviews))

print("\n===== TOTAL POSITIVE REVIEWS =====")
positive = (reviews["sentiment"].str.lower() == "positive").sum()
print(positive)

print("\n===== TOTAL NEGATIVE REVIEWS =====")
negative = (reviews["sentiment"].str.lower() == "negative").sum()
print(negative)

print("\n===== TOTAL NEUTRAL REVIEWS =====")
neutral = (reviews["sentiment"].str.lower() == "neutral").sum()
print(neutral)

print("\n===== SENTIMENT PERCENTAGE =====")

total = len(reviews)

print(f"Positive : {positive} ({positive/total*100:.2f}%)")
print(f"Negative : {negative} ({negative/total*100:.2f}%)")
print(f"Neutral  : {neutral} ({neutral/total*100:.2f}%)")
print("\n===== AVERAGE HELPFUL VOTES =====")
print(reviews["helpful_votes"].mean())

print("\n===== MAX HELPFUL VOTES =====")
print(reviews["helpful_votes"].max())

print("\n===== MIN HELPFUL VOTES =====")
print(reviews["helpful_votes"].min())
###=========subscribers=========####
# ==========================================
# SUBSCRIBERS DATA ANALYSIS
# ==========================================

print("\n==============================")
print("===== SUBSCRIBERS DATA =====")
print("==============================")

print("\n===== COMPLETE DATA =====")
print(subscribers)

print("\n===== FIRST 5 ROWS =====")
print(subscribers.head())

print("\n===== LAST 5 ROWS =====")
print(subscribers.tail())

print("\n===== RANDOM 5 ROWS =====")
print(subscribers.sample(5))

print("\n===== COLUMN NAMES =====")
print(subscribers.columns)

print("\n===== DATA TYPES =====")
print(subscribers.dtypes)

print("\n===== SHAPE (ROWS, COLUMNS) =====")
print(subscribers.shape)

print("\n===== NUMBER OF ROWS =====")
print(subscribers.shape[0])

print("\n===== NUMBER OF COLUMNS =====")
print(subscribers.shape[1])

print("\n===== MISSING VALUES =====")
print(subscribers.isnull().sum())

print("\n===== TOTAL MISSING VALUES =====")
print(subscribers.isnull().sum().sum())

print("\n===== DUPLICATE VALUES =====")
print(subscribers.duplicated().sum())

print("\n===== UNIQUE VALUES =====")
print(subscribers.nunique())

print("\n===== STATISTICAL SUMMARY =====")
print(pd.DataFrame({
    "Count": subscribers.count(),
    "Sum": subscribers.sum(numeric_only=True),
    "Mean": subscribers.mean(numeric_only=True),
    "Median": subscribers.median(numeric_only=True),
    "Min": subscribers.min(numeric_only=True),
    "Max": subscribers.max(numeric_only=True),
    "Std": subscribers.std(numeric_only=True),
    "Variance": subscribers.var(numeric_only=True)
}))

print("\n===== DESCRIBE =====")
print(subscribers.describe(include="all"))

print("\n===== CORRELATION =====")
print(subscribers.corr(numeric_only=True))
# ==========================================
# SUBSCRIBERS SUMMARY
# ==========================================

print("\n===== TOTAL SUBSCRIBERS =====")
print(len(subscribers))

print("\n===== ACTIVE SUBSCRIBERS =====")
active = subscribers["is_active"].sum()
print(active)

print("\n===== INACTIVE SUBSCRIBERS =====")
inactive = (~subscribers["is_active"]).sum()
print(inactive)

print("\n===== ACTIVE PERCENTAGE =====")
print(f"{active/len(subscribers)*100:.2f}%")

print("\n===== PLAN TYPE =====")
print(subscribers["plan_type"].value_counts())

print("\n===== COUNTRY =====")
print(subscribers["country"].value_counts())

print("\n===== REGION =====")
print(subscribers["region"].value_counts())

print("\n===== GENDER =====")
print(subscribers["gender"].value_counts())

print("\n===== DEVICE =====")
print(subscribers["primary_device"].value_counts())

print("\n===== PAYMENT METHOD =====")
print(subscribers["payment_method"].value_counts())

###=========titles========#####

# ==========================================
# TITLES DATA ANALYSIS
# ==========================================

print("\n==============================")
print("===== TITLES DATA =====")
print("==============================")

print("\n===== COMPLETE DATA =====")
print(titles)

print("\n===== FIRST 5 ROWS =====")
print(titles.head())

print("\n===== LAST 5 ROWS =====")
print(titles.tail())

print("\n===== RANDOM 5 ROWS =====")
print(titles.sample(5))

print("\n===== COLUMN NAMES =====")
print(titles.columns)

print("\n===== DATA TYPES =====")
print(titles.dtypes)

print("\n===== SHAPE (ROWS, COLUMNS) =====")
print(titles.shape)

print("\n===== NUMBER OF ROWS =====")
print(titles.shape[0])

print("\n===== NUMBER OF COLUMNS =====")
print(titles.shape[1])

print("\n===== MISSING VALUES =====")
print(titles.isnull().sum())

print("\n===== TOTAL MISSING VALUES =====")
print(titles.isnull().sum().sum())

print("\n===== DUPLICATE VALUES =====")
print(titles.duplicated().sum())

print("\n===== UNIQUE VALUES =====")
print(titles.nunique())

print("\n===== STATISTICAL SUMMARY =====")
print(pd.DataFrame({
    "Count": titles.count(),
    "Sum": titles.sum(numeric_only=True),
    "Mean": titles.mean(numeric_only=True),
    "Median": titles.median(numeric_only=True),
    "Min": titles.min(numeric_only=True),
    "Max": titles.max(numeric_only=True),
    "Std": titles.std(numeric_only=True),
    "Variance": titles.var(numeric_only=True)
}))

print("\n===== DESCRIBE =====")
print(titles.describe(include="all"))

print("\n===== CORRELATION =====")
print(titles.corr(numeric_only=True))
# ==========================================
# TITLES SUMMARY
# ==========================================

print("\n===== TOTAL TITLES =====")
print(len(titles))

print("\n===== MOVIES =====")
movies = (titles["type"]=="Movie").sum()
print(movies)

print("\n===== TV SHOWS =====")
shows = (titles["type"]=="TV Show").sum()
print(shows)

print("\n===== ORIGINAL CONTENT =====")
original = titles["is_original"].sum()
print(original)

print("\n===== LICENSED CONTENT =====")
licensed = (~titles["is_original"]).sum()
print(licensed)

print("\n===== GENRES =====")
print(titles["primary_genre"].value_counts())

print("\n===== LANGUAGES =====")
print(titles["language"].value_counts())

print("\n===== RELEASE YEAR =====")
print(titles["release_year"].value_counts().head(10))

####==========watch_history===========

# ==========================================
# WATCH_HISTORY DATA ANALYSIS
# ==========================================

print("\n==============================")
print("===== WATCH HISTORY DATA =====")
print("==============================")

print("\n===== COMPLETE DATA =====")
print(watch_history)

print("\n===== FIRST 5 ROWS =====")
print(watch_history.head())

print("\n===== LAST 5 ROWS =====")
print(watch_history.tail())

print("\n===== RANDOM 5 ROWS =====")
print(watch_history.sample(5))

print("\n===== COLUMN NAMES =====")
print(watch_history.columns)

print("\n===== DATA TYPES =====")
print(watch_history.dtypes)

print("\n===== SHAPE (ROWS, COLUMNS) =====")
print(watch_history.shape)

print("\n===== NUMBER OF ROWS =====")
print(watch_history.shape[0])

print("\n===== NUMBER OF COLUMNS =====")
print(watch_history.shape[1])

print("\n===== MISSING VALUES =====")
print(watch_history.isnull().sum())

print("\n===== TOTAL MISSING VALUES =====")
print(watch_history.isnull().sum().sum())

print("\n===== DUPLICATE VALUES =====")
print(watch_history.duplicated().sum())

print("\n===== UNIQUE VALUES =====")
print(watch_history.nunique())

print("\n===== STATISTICAL SUMMARY =====")
print(pd.DataFrame({
    "Count": watch_history.count(),
    "Sum": watch_history.sum(numeric_only=True),
    "Mean": watch_history.mean(numeric_only=True),
    "Median": watch_history.median(numeric_only=True),
    "Min": watch_history.min(numeric_only=True),
    "Max": watch_history.max(numeric_only=True),
    "Std": watch_history.std(numeric_only=True),
    "Variance": watch_history.var(numeric_only=True)
}))

print("\n===== DESCRIBE =====")
print(watch_history.describe(include="all"))

print("\n===== CORRELATION =====")
print(watch_history.corr(numeric_only=True))
# ==========================================
# WATCH HISTORY SUMMARY
# ==========================================

print("\n===== TOTAL WATCHES =====")
print(len(watch_history))

print("\n===== TOTAL WATCH HOURS =====")
print(round(watch_history["watch_duration_min"].sum()/60,2))

print("\n===== AVERAGE WATCH DURATION =====")
print(round(watch_history["watch_duration_min"].mean(),2))

print("\n===== AVERAGE COMPLETION =====")
print(round(watch_history["completion_pct"].mean(),2))

print("\n===== COMPLETED WATCHES =====")
completed = watch_history["completed"].sum()
print(completed)

print("\n===== INCOMPLETE WATCHES =====")
incomplete = (~watch_history["completed"]).sum()
print(incomplete)

print("\n===== DEVICE USAGE =====")
print(watch_history["device"].value_counts())

print("\n===== REGION =====")
print(watch_history["region"].value_counts())

####========watchlist==========#######

# ==========================================
# WATCHLIST DATA ANALYSIS
# ==========================================

print("\n==============================")
print("===== WATCHLIST DATA =====")
print("==============================")

print("\n===== COMPLETE DATA =====")
print(watchlist)

print("\n===== FIRST 5 ROWS =====")
print(watchlist.head())

print("\n===== LAST 5 ROWS =====")
print(watchlist.tail())

print("\n===== RANDOM 5 ROWS =====")
print(watchlist.sample(5))

print("\n===== COLUMN NAMES =====")
print(watchlist.columns)

print("\n===== DATA TYPES =====")
print(watchlist.dtypes)

print("\n===== SHAPE (ROWS, COLUMNS) =====")
print(watchlist.shape)

print("\n===== NUMBER OF ROWS =====")
print(watchlist.shape[0])

print("\n===== NUMBER OF COLUMNS =====")
print(watchlist.shape[1])

print("\n===== MISSING VALUES =====")
print(watchlist.isnull().sum())

print("\n===== TOTAL MISSING VALUES =====")
print(watchlist.isnull().sum().sum())

print("\n===== DUPLICATE VALUES =====")
print(watchlist.duplicated().sum())

print("\n===== UNIQUE VALUES =====")
print(watchlist.nunique())

print("\n===== STATISTICAL SUMMARY =====")
print(pd.DataFrame({
    "Count": watchlist.count(),
    "Sum": watchlist.sum(numeric_only=True),
    "Mean": watchlist.mean(numeric_only=True),
    "Median": watchlist.median(numeric_only=True),
    "Min": watchlist.min(numeric_only=True),
    "Max": watchlist.max(numeric_only=True),
    "Std": watchlist.std(numeric_only=True),
    "Variance": watchlist.var(numeric_only=True)
}))

print("\n===== DESCRIBE =====")
print(watchlist.describe(include="all"))

print("\n===== CORRELATION =====")
print(watchlist.corr(numeric_only=True))
# ==========================================
# WATCHLIST SUMMARY
# ==========================================

print("\n===== TOTAL WATCHLIST =====")
print(len(watchlist))

print("\n===== WATCHED =====")
watched = watchlist["watched"].sum()
print(watched)

print("\n===== NOT WATCHED =====")
not_watched = (~watchlist["watched"]).sum()
print(not_watched)

print("\n===== WATCHLIST CONVERSION =====")
print(f"{watched/len(watchlist)*100:.2f}%")
# =====================================================
# BUSINESS KPI ANALYSIS
# =====================================================

print("\n" + "="*60)
print("           STREAMFLIX BUSINESS KPI")
print("="*60)

# =====================================================
# BUSINESS KPI ANALYSIS
# =====================================================

print("\n" + "="*60)
print("           STREAMFLIX BUSINESS KPI")
print("="*60)

# Total Watch Hours
total_watch_hours = watch_history["watch_duration_min"].sum() / 60
print(f"\nTotal Watch Hours : {total_watch_hours:,.2f}")

# Average Watch Time
avg_watch = watch_history["watch_duration_min"].mean()
print(f"Average Watch Time (Minutes) : {avg_watch:.2f}")

# Average Completion
avg_completion = watch_history["completion_pct"].mean()
print(f"Average Completion % : {avg_completion:.2f}")

# Total Views
print(f"Total Views : {len(watch_history):,}")

# Completed Views
completed = watch_history["completed"].sum()
print(f"Completed Views : {completed}")

# Completion Rate
completion_rate = (completed / len(watch_history)) * 100
print(f"Completion Rate : {completion_rate:.2f}%")


# Total Subscribers
total_subscribers = subscribers["subscriber_id"].nunique()
print(f"Total Subscribers : {total_subscribers}")

# Active Subscribers
active = subscribers["is_active"].sum()
inactive = (~subscribers["is_active"]).sum()

print(f"Active Subscribers : {active}")
print(f"Inactive Subscribers : {inactive}")

# Active Rate
print(f"Active Rate : {(active/total_subscribers)*100:.2f}%")

# Churn Rate
print(f"Churn Rate : {(inactive/total_subscribers)*100:.2f}%")

# Total Titles
print(f"Total Titles : {titles['title_id'].nunique()}")

# Movies
movies = (titles["type"]=="Movie").sum()
print(f"Movies : {movies}")

# TV Shows
shows = (titles["type"]=="TV Show").sum()
print(f"TV Shows : {shows}")

# Originals
original = titles["is_original"].sum()
print(f"Original Content : {original}")

# Licensed
licensed = (~titles["is_original"]).sum()
print(f"Licensed Content : {licensed}")

# Average Rating
print(f"Average Rating : {df['rating'].mean():.2f}")

# Highest Rating
print(f"Highest Rating : {df['rating'].max()}")

# Lowest Rating
print(f"Lowest Rating : {df['rating'].min()}")

# Total Reviews
print(f"Total Reviews : {len(reviews)}")

# Watchlist
print(f"Total Watchlist : {len(watchlist)}")

# Watchlist Converted
converted = watchlist["watched"].sum()

print(f"Watchlist Converted : {converted}")

print(f"Conversion Rate : {(converted/len(watchlist))*100:.2f}%")

print("="*60)
# =====================================================
# MERGE TABLES
# =====================================================

merged = watch_history.merge(
    subscribers,
    on="subscriber_id",
    how="left"
)

merged = merged.merge(
    titles,
    on="title_id",
    how="left"
) # ==============================
# FIX MERGED COLUMN NAMES
# ==============================

print("\nMerged Columns Before Fix:")
print(merged.columns.tolist())


# Country Fix
if "country" not in merged.columns:

    if "country_x" in merged.columns:
        merged["country"] = merged["country_x"]

    elif "country_y" in merged.columns:
        merged["country"] = merged["country_y"]

    else:
        merged["country"] = "Unknown"


# Plan Fix
if "plan_type" not in merged.columns:

    if "plan_type_x" in merged.columns:
        merged["plan_type"] = merged["plan_type_x"]

    elif "plan_type_y" in merged.columns:
        merged["plan_type"] = merged["plan_type_y"]


# Language Fix
if "language" not in merged.columns:

    if "language_x" in merged.columns:
        merged["language"] = merged["language_x"]

    elif "language_y" in merged.columns:
        merged["language"] = merged["language_y"]


print("\nMerged Columns After Fix:")
print(merged.columns.tolist())


print("\nMerged Shape :", merged.shape)
print(merged.head())
# =====================================================
# ADVANCED CHARTS
# =====================================================

# Top 10 Genres by Watch Hours

genre = merged.groupby("primary_genre")["watch_duration_min"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
genre.plot(kind="bar",color="orange")
plt.title("Top 10 Genres By Watch Hours")
plt.xlabel("Genre")
plt.ylabel("Minutes Watched")
plt.xticks(rotation=45)
plt.grid(True)
plt.close()




# Top 10 Countries

print("Merged columns:")
print(merged.columns.tolist())

country = merged.groupby("country")["watch_duration_min"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
country.plot(kind="bar",color="green")
plt.title("Top 10 Countries By Watch Hours")
plt.xlabel("Country")
plt.ylabel("Minutes Watched")
plt.xticks(rotation=45)
plt.grid(True)
plt.close()




# Plan Wise Watch Time

plan = merged.groupby("plan_type")["watch_duration_min"].sum()/60

plt.figure(figsize=(7,5))
plan.plot(kind="pie",autopct="%1.1f%%")
plt.title("Plan Wise Watch Hours")
plt.ylabel("")



# Device Usage

device = merged["device"].value_counts()

plt.figure(figsize=(7,5))
device.plot(kind="bar",color="red")
plt.title("Device Usage")
plt.xlabel("Device")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.grid(True)
plt.close()




# Language Distribution

language = merged["language"].value_counts().head(10)

plt.figure(figsize=(10,5))
language.plot(kind="bar",color="purple")
plt.title("Top Languages")
plt.xlabel("Language")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True)
plt.close()

# =====================================================
# FINAL BUSINESS INSIGHTS
# =====================================================

print("\n" + "="*70)
print("               STREAMFLIX FINAL REPORT")
print("="*70)

print(f"Total Subscribers        : {subscribers['subscriber_id'].nunique()}")
print(f"Total Titles             : {titles['title_id'].nunique()}")
print(f"Total Ratings            : {len(df)}")
print(f"Total Reviews            : {len(reviews)}")
print(f"Total Watch Sessions     : {len(watch_history)}")
print(f"Total Watchlist Entries  : {len(watchlist)}")

print("-"*70)

print(f"Average Rating           : {df['rating'].mean():.2f}")
print(f"Average Watch Time       : {watch_history['watch_duration_min'].mean():.2f} Minutes")
print(f"Average Completion       : {watch_history['completion_pct'].mean():.2f}%")
print(f"Total Watch Hours        : {watch_history['watch_duration_min'].sum()/60:.2f}")

print("-"*70)

print("Top Country :")
print(subscribers["country"].value_counts().head(1))

print("\nTop Genre :")
print(titles["primary_genre"].value_counts().head(1))

print("\nMost Used Device :")
print(watch_history["device"].value_counts().head(1))

print("\nMost Popular Plan :")
print(subscribers["plan_type"].value_counts().head(1))

print("\nMost Common Language :")
print(titles["language"].value_counts().head(1))

print("="*70)
# =====================================================
# ADVANCED DATA ANALYSIS
# =====================================================

# -----------------------------------------------------
# MERGE TABLES
# -----------------------------------------------------
# =====================================================
# MERGE TABLES (FIXED)
# =====================================================

merged = watch_history.merge(
    subscribers,
    on="subscriber_id",
    how="left",
    suffixes=("", "_subscriber")
)

merged = merged.merge(
    titles,
    on="title_id",
    how="left",
    suffixes=("", "_title")
)

# Check columns after merge
print("\nMerged Shape :", merged.shape)

print("\nMerged Columns:")
print(merged.columns.tolist())

print("\nMerged Data:")
print(merged.head())


# Fix country column name
if "country" not in merged.columns:

    if "country_subscriber" in merged.columns:
        merged.rename(
            columns={"country_subscriber":"country"},
            inplace=True
        )

    elif "country_x" in merged.columns:
        merged.rename(
            columns={"country_x":"country"},
            inplace=True
        )

    elif "country_y" in merged.columns:
        merged.rename(
            columns={"country_y":"country"},
            inplace=True
        )

print("\nCountry column check:")
print("country" in merged.columns)


# -----------------------------------------------------
# BUSINESS KPI
# -----------------------------------------------------

print("\n" + "="*60)
print("            STREAMFLIX BUSINESS KPI")
print("="*60)

print("Total Watch Hours :", round(merged["watch_duration_min"].sum()/60,2))

print("Average Watch Time :", round(merged["watch_duration_min"].mean(),2))

print("Average Completion :", round(merged["completion_pct"].mean(),2))

print("Total Subscribers :", subscribers["subscriber_id"].nunique())

active = subscribers["is_active"].sum()
inactive = (~subscribers["is_active"]).sum()

print("Active Subscribers :", active)
print("Inactive Subscribers :", inactive)

print("Active Rate : {:.2f}%".format((active/len(subscribers))*100))

print("Churn Rate : {:.2f}%".format((inactive/len(subscribers))*100))

print("Total Titles :", titles["title_id"].nunique())

print("Movies :", (titles["type"]=="Movie").sum())

print("TV Shows :", (titles["type"]=="TV Show").sum())

print("Original Content :", titles["is_original"].sum())

print("Licensed Content :", (~titles["is_original"]).sum())

print("Average Rating :", round(df["rating"].mean(),2))

print("Highest Rating :", df["rating"].max())

print("Lowest Rating :", df["rating"].min())

print("Total Reviews :", len(reviews))

print("Total Watchlist :", len(watchlist))

converted = watchlist["watched"].sum()

print("Watchlist Converted :", converted)

print("Conversion Rate : {:.2f}%".format((converted/len(watchlist))*100))

print("="*60)

# -----------------------------------------------------
# TOP GENRES
# -----------------------------------------------------

genre = merged.groupby("primary_genre")["watch_duration_min"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
genre.plot(kind="bar",color="orange")
plt.title("Top 10 Genres by Watch Hours")
plt.xlabel("Genre")
plt.ylabel("Minutes")
plt.xticks(rotation=45)
plt.grid(True)
plt.close()



# -----------------------------------------------------
# TOP COUNTRIES
# -----------------------------------------------------

country = merged.groupby("country")["watch_duration_min"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
country.plot(kind="bar",color="green")
plt.title("Top 10 Countries by Watch Hours")
plt.xlabel("Country")
plt.ylabel("Minutes")
plt.xticks(rotation=45)
plt.grid(True)
plt.close()


# -----------------------------------------------------
# PLAN WISE WATCH HOURS
# -----------------------------------------------------

plan = merged.groupby("plan_type")["watch_duration_min"].sum()/60

plt.figure(figsize=(7,5))
plan.plot(kind="pie",autopct="%1.1f%%")
plt.title("Plan Wise Watch Hours")
plt.ylabel("")


# -----------------------------------------------------
# DEVICE USAGE
# -----------------------------------------------------

device = merged["device"].value_counts()

plt.figure(figsize=(8,5))
device.plot(kind="bar",color="red")
plt.title("Device Usage")
plt.xlabel("Device")
plt.ylabel("Count")
plt.grid(True)



# -----------------------------------------------------
# TOP LANGUAGES
# -----------------------------------------------------

language = merged["language"].value_counts().head(10)

plt.figure(figsize=(10,5))
language.plot(kind="bar",color="purple")
plt.title("Top Languages")
plt.xlabel("Language")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True)
plt.close()



# -----------------------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    watch_history.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")


# -----------------------------------------------------
# OUTLIER DETECTION
# -----------------------------------------------------

plt.figure(figsize=(8,5))
plt.boxplot(watch_history["watch_duration_min"])
plt.title("Watch Duration Outlier Detection")
plt.ylabel("Minutes")


# -----------------------------------------------------
# SCATTER PLOT
# -----------------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    watch_history["watch_duration_min"],
    watch_history["completion_pct"],
    alpha=0.3
)

plt.xlabel("Watch Duration")
plt.ylabel("Completion %")
plt.title("Watch Duration vs Completion")
plt.grid(True)
plt.close()



# -----------------------------------------------------
# MONTHLY WATCH TREND
# -----------------------------------------------------

watch_history["watch_date"] = pd.to_datetime(watch_history["watch_date"])

monthly = watch_history.groupby(
    watch_history["watch_date"].dt.to_period("M")
).size()

plt.figure(figsize=(12,5))
monthly.plot()
plt.title("Monthly Viewing Trend")
plt.xlabel("Month")
plt.ylabel("Sessions")
plt.grid(True)
plt.close()



# -----------------------------------------------------
# MONTHLY WATCH HOURS
# -----------------------------------------------------

monthly_hours = watch_history.groupby(
    watch_history["watch_date"].dt.to_period("M")
)["watch_duration_min"].sum()/60

plt.figure(figsize=(12,5))
monthly_hours.plot(color="green")
plt.title("Monthly Watch Hours")
plt.xlabel("Month")
plt.ylabel("Hours")
plt.grid(True)
plt.close()



# -----------------------------------------------------
# TOP 10 MOST WATCHED TITLES
# -----------------------------------------------------

top_titles = merged.groupby("title_name")["watch_duration_min"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,5))
top_titles.plot(kind="bar",color="blue")
plt.title("Top 10 Most Watched Titles")
plt.xlabel("Title")
plt.ylabel("Minutes")
plt.xticks(rotation=45)
plt.grid(True)
plt.close()



# -----------------------------------------------------
# REVENUE BY PLAN
# -----------------------------------------------------

revenue = subscribers.groupby("plan_type")["monthly_price_usd"].sum()

plt.figure(figsize=(7,5))
revenue.plot(kind="bar",color="brown")
plt.title("Revenue by Plan")
plt.xlabel("Plan")
plt.ylabel("Revenue ($)")
plt.grid(True)
plt.close("all")



# -----------------------------------------------------
# FINAL PROJECT SUMMARY
# -----------------------------------------------------

print("\n" + "="*70)
print("             STREAMFLIX FINAL SUMMARY")
print("="*70)

print("Total Subscribers :", subscribers.shape[0])
print("Total Titles :", titles.shape[0])
print("Total Ratings :", df.shape[0])
print("Total Reviews :", reviews.shape[0])
print("Total Watch Sessions :", watch_history.shape[0])
print("Total Watchlist :", watchlist.shape[0])

print("-"*70)

print("Average Rating :", round(df["rating"].mean(),2))
print("Average Completion :", round(watch_history["completion_pct"].mean(),2))
print("Total Watch Hours :", round(watch_history["watch_duration_min"].sum()/60,2))

print("="*70)

print("\n" + "="*60)
print("MACHINE LEARNING - CHURN PREDICTION")
print("="*60)

# Copy Dataset
ml_data = subscribers.copy()

# Remove Missing Values
ml_data = ml_data.dropna()

# Encode Categorical Columns
le = LabelEncoder()

categorical_columns = [
    "country",
    "region",
    "gender",
    "plan_type",
    "primary_device",
    "payment_method"
]

for col in categorical_columns:
    ml_data[col] = le.fit_transform(ml_data[col].astype(str))

# Convert Boolean
ml_data["is_active"] = ml_data["is_active"].astype(int)

# Features
X = ml_data[[
    "age",
    "monthly_price_usd",
    "household_size",
    "tenure_months",
    "country",
    "region",
    "gender",
    "plan_type",
    "primary_device",
    "payment_method"
]]

# Target
y = ml_data["is_active"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(acc*100,2),"%")

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))

# Feature Importance
importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# Feature Importance Chart

plt.figure(figsize=(10,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.grid(True)




print("="*60)
print("Machine Learning Completed Successfully")
print("="*60)
print("\n" + "="*70)
print("LINEAR REGRESSION - WATCH DURATION PREDICTION")
print("="*70)

# -----------------------------------------------------
# COPY DATA
# -----------------------------------------------------

lr_data = watch_history.copy()

# Remove Missing Values
lr_data = lr_data.dropna()

# -----------------------------------------------------
# TARGET COLUMN
# -----------------------------------------------------

target = "watch_duration_min"

# -----------------------------------------------------
# ENCODE ALL OBJECT COLUMNS
# -----------------------------------------------------

encoder = LabelEncoder()

for col in lr_data.select_dtypes(include=["object", "string"]).columns:
    lr_data[col] = encoder.fit_transform(lr_data[col].astype(str))


# -----------------------------------------------------
# CONVERT BOOLEAN TO INTEGER
# -----------------------------------------------------

for col in lr_data.select_dtypes(include="bool").columns:
    lr_data[col] = lr_data[col].astype(int)
# -----------------------------------------------------
# CONVERT DATETIME TO INTEGER
# -----------------------------------------------------

for col in lr_data.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
    lr_data[col] = pd.to_datetime(lr_data[col])
    lr_data[col] = lr_data[col].astype("int64")
    
    

# -----------------------------------------------------
# FEATURES
# -----------------------------------------------------

X = lr_data.drop(columns=[target])

# Remove ID Columns (Optional)

remove_cols = []

for c in X.columns:
    if c.lower().endswith("_id"):
        remove_cols.append(c)

X = X.drop(columns=remove_cols)

# -----------------------------------------------------
# TARGET
# -----------------------------------------------------

y = lr_data[target]

# -----------------------------------------------------
# TRAIN TEST SPLIT
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------------------------------
# MODEL
# -----------------------------------------------------

model = LinearRegression()

print("Remaining Datatypes:")
print(X_train.dtypes)

model.fit(X_train, y_train)


# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------

y_pred = model.predict(X_test)

# -----------------------------------------------------
# MODEL EVALUATION
# -----------------------------------------------------

print("\nModel Performance")
print("-"*40)

print("R2 Score :", round(r2_score(y_test,y_pred),4))

print("MAE :", round(mean_absolute_error(y_test,y_pred),2))

rmse = mean_squared_error(y_test,y_pred)**0.5

print("RMSE :", round(rmse,2))

# -----------------------------------------------------
# COEFFICIENT TABLE
# -----------------------------------------------------

coef = pd.DataFrame({
    "Feature":X.columns,
    "Coefficient":model.coef_
})

coef = coef.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Coefficients")
print(coef)

# -----------------------------------------------------
# SAMPLE PREDICTIONS
# -----------------------------------------------------

result = pd.DataFrame({
    "Actual":y_test.values,
    "Predicted":y_pred
})

print("\nActual vs Predicted")
print(result.head(15))

# -----------------------------------------------------
# ACTUAL VS PREDICTED GRAPH
# -----------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()],
    color="red"
)

plt.title("Actual vs Predicted Watch Duration")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.grid(True)
plt.close()




# -----------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------

plt.figure(figsize=(12,5))

coef.sort_values(
    by="Coefficient"
).plot(
    x="Feature",
    y="Coefficient",
    kind="barh",
    legend=False
)

plt.title("Feature Importance (Linear Regression)")

plt.xlabel("Coefficient")

plt.grid(True)
plt.close()



# -----------------------------------------------------
# RESIDUAL ERROR PLOT
# -----------------------------------------------------

residual = y_test - y_pred

plt.figure(figsize=(8,5))

plt.scatter(
    y_pred,
    residual,
    alpha=0.5
)

plt.axhline(
    y=0,
    color="red"
)

plt.title("Residual Error Plot")

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.grid(True)
plt.close()




print("="*70)
print("LINEAR REGRESSION COMPLETED SUCCESSFULLY")
print("="*70)
print("\n" + "="*70)
print("RATING PREDICTION USING RANDOM FOREST REGRESSOR")
print("="*70)

# -----------------------------------------------------
# MERGE RATINGS WITH WATCH HISTORY
# -----------------------------------------------------
# -----------------------------------------------------
# MERGE RATINGS WITH WATCH HISTORY
# -----------------------------------------------------

print("df columns:")
print(df.columns.tolist())

print("\nwatch_history columns:")
print(watch_history.columns.tolist())

rating_data = df.merge(
    watch_history,
    on=["subscriber_id", "title_id"],
    how="inner"
)



rating_data = rating_data.dropna()



# -----------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# -----------------------------------------------------

encoder = LabelEncoder()
for col in rating_data.select_dtypes(include=["object", "string"]).columns:



    rating_data[col] = encoder.fit_transform(
        rating_data[col].astype(str)
    )

# -----------------------------------------------------
# CONVERT BOOLEAN TO INTEGER
# -----------------------------------------------------

for col in rating_data.select_dtypes(include="bool").columns:
    rating_data[col] = rating_data[col].astype(int)
# -----------------------------------------------------
# CONVERT DATETIME COLUMNS
# -----------------------------------------------------

for col in rating_data.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
    rating_data[col] = pd.to_datetime(rating_data[col])
    rating_data[col] = rating_data[col].astype("int64")
    

# -----------------------------------------------------
# TARGET
# -----------------------------------------------------

target = "rating"

# -----------------------------------------------------
# FEATURES
# -----------------------------------------------------

X = rating_data.drop(columns=[target])

# Remove ID Columns
drop_cols = [c for c in X.columns if c.lower().endswith("_id")]
X = X.drop(columns=drop_cols)

y = rating_data[target]

# -----------------------------------------------------
# TRAIN TEST SPLIT
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------------------------------
# MODEL
# -----------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------

y_pred = model.predict(X_test)

# -----------------------------------------------------
# EVALUATION
# -----------------------------------------------------

print("\nMODEL PERFORMANCE")
print("-"*50)

print("R2 Score :", round(r2_score(y_test, y_pred),4))
print("MAE :", round(mean_absolute_error(y_test, y_pred),2))
print("RMSE :", round(mean_squared_error(y_test, y_pred)**0.5,2))

# -----------------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP IMPORTANT FEATURES")
print(importance)

# -----------------------------------------------------
# SAMPLE PREDICTIONS
# -----------------------------------------------------

prediction = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": y_pred
})

print("\nACTUAL VS PREDICTED")
print(prediction.head(10))


# -----------------------------------------------------
# FEATURE IMPORTANCE GRAPH
# -----------------------------------------------------

plt.figure(figsize=(10,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.title("Feature Importance For Rating Prediction")

plt.xlabel("Features")
plt.ylabel("Importance")

plt.grid(True)
plt.close()



# -----------------------------------------------------
# ACTUAL VS PREDICTED
# -----------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("Actual vs Predicted Rating")

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

plt.grid(True)
plt.close("all")
plt.close()



print("="*70)
print("RATING PREDICTION COMPLETED SUCCESSFULLY")
print("="*70)
print("\n" + "="*70)
print("        MOVIE RECOMMENDATION SYSTEM")
print("="*70)
# ===========================================
# ==========================================================
# CREATE COPY OF DATA
# -----------------------------------------------------

movies = titles.copy()

# Remove Missing Values
movies = movies.dropna(subset=["title_name", "primary_genre"])

# Keep Required Columns
movies = movies[["title_name", "primary_genre"]]

# Remove Duplicate Titles
movies = movies.drop_duplicates(subset="title_name")

# -----------------------------------------------------
# CREATE GENRE MATRIX
# -----------------------------------------------------

cv = CountVectorizer()

genre_matrix = cv.fit_transform(movies["primary_genre"])

# -----------------------------------------------------
# CALCULATE SIMILARITY
# -----------------------------------------------------

similarity = cosine_similarity(genre_matrix)

# -----------------------------------------------------
# RECOMMEND FUNCTION
# -----------------------------------------------------

def recommend_movie(movie_name, top_n=5):

    movie_name = movie_name.lower()

    movie_index = movies[
        movies["title_name"].str.lower() == movie_name
    ].index

    if len(movie_index) == 0:
        print("\nMovie Not Found!")
        return

    movie_index = movie_index[0]

    distance = list(enumerate(similarity[movie_index]))

    distance = sorted(
        distance,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommended Movies")
    print("-"*40)

    count = 0

    for i in distance[1:]:

        print(movies.iloc[i[0]]["title_name"])

        count += 1

        if count == top_n:
            break

# -----------------------------------------------------
# SHOW SAMPLE MOVIES
# -----------------------------------------------------

print("\nAvailable Movies")

print("-"*40)

print(movies["title_name"].head(10))

# -----------------------------------------------------
# USER INPUT
# -----------------------------------------------------

movie = input("\nEnter Movie Name : ")

recommend_movie(movie)

print("="*70)
print("RECOMMENDATION COMPLETED")
print("="*70)




# ==========================================================
# STREAMFLIX — 3 UNIQUE PROFESSIONAL DASHBOARDS
# ==========================================================
# Dashboard 1 : Executive / Business Overview
# Dashboard 2 : Content / User Intelligence
# Dashboard 3 : AI / Machine Learning Lab
#
# Each dashboard opens in its own clean window.
# No individual charts are opened separately.
# ==========================================================

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

plt.close("all")

# ==========================================================
# PREMIUM THEME
# ==========================================================

BG = "#07111F"
PANEL = "#0F1C2E"
PANEL2 = "#13243A"
BORDER = "#263A55"

WHITE = "#F8FAFC"
MUTED = "#94A3B8"

CYAN = "#22D3EE"
BLUE = "#60A5FA"
PURPLE = "#A78BFA"
PINK = "#F472B6"
ORANGE = "#F59E0B"
GREEN = "#34D399"
RED = "#FB7185"
GOLD = "#FACC15"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
    "font.size": 9
})

# ==========================================================
# COMMON FUNCTIONS
# ==========================================================

def style_ax(ax):
    ax.set_facecolor(PANEL)

    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(BORDER)
        s.set_linewidth(0.8)

    ax.grid(
        axis="y",
        color=WHITE,
        alpha=0.07,
        linewidth=0.7
    )

    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8)

def chart_title(ax, title, subtitle="", accent=CYAN):
    style_ax(ax)

    ax.set_title(
        title,
        loc="center",
        color=WHITE,
        fontsize=12,
        fontweight="bold",
        pad=19
    )

    ax.text(
        0.5, 1.035,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=7.5
    )

    ax.plot(
        [0.40, 0.60],
        [1.015, 1.015],
        transform=ax.transAxes,
        color=accent,
        linewidth=2.2,
        solid_capstyle="round",
        clip_on=False
    )

def header(fig, title, subtitle, section_no, accent):
    fig.text(
        0.50, 0.965,
        title,
        ha="center",
        va="top",
        fontsize=25,
        fontweight="bold",
        color=WHITE
    )

    fig.text(
        0.50, 0.930,
        subtitle,
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED
    )

    fig.text(
        0.035, 0.965,
        f"0{section_no}",
        ha="left",
        va="top",
        fontsize=14,
        color=accent,
        fontweight="bold"
    )

    fig.text(
        0.965, 0.965,
        "STREAMFLIX  •  ANALYTICS",
        ha="right",
        va="top",
        fontsize=8.5,
        color=accent,
        fontweight="bold"
    )

    fig.lines.append(
        plt.Line2D(
            [0.035, 0.965],
            [0.900, 0.900],
            transform=fig.transFigure,
            color=BORDER,
            linewidth=1
        )
    )

def kpi(fig, x, value, label, accent):
    ax = fig.add_axes([x, 0.825, 0.165, 0.060])

    ax.set_facecolor(PANEL2)
    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_color(accent)
        s.set_linewidth(1)

    ax.text(
        0.08, 0.62,
        str(value),
        transform=ax.transAxes,
        color=accent,
        fontsize=18,
        fontweight="bold",
        va="center"
    )

    ax.text(
        0.08, 0.23,
        label,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7.5,
        va="center"
    )

def footer(fig, text, accent=CYAN):
    fig.lines.append(
        plt.Line2D(
            [0.035, 0.965],
            [0.035, 0.035],
            transform=fig.transFigure,
            color=BORDER,
            linewidth=1
        )
    )

    fig.text(
        0.50, 0.018,
        text,
        ha="center",
        va="center",
        fontsize=8,
        color=accent,
        fontweight="bold"
    )

def short(values, n=18):
    return [str(v)[:n] for v in values]

# ==========================================================
# PREPARE DATA
# ==========================================================

try:
    monthly_dash = watch_history.groupby(
        watch_history["watch_date"].dt.to_period("M")
    )["watch_duration_min"].sum() / 60
except Exception:
    monthly_dash = pd.Series(dtype=float)

try:
    # Platform top 8 titles remain the same chart.
    # If a movie is selected, move that movie to the TOP position
    # without changing the chart size/style.
    selected_chart_title = None

    platform_top = merged.groupby(
        "title_name"
    )["watch_duration_min"].sum().sort_values(
        ascending=False
    ).head(8)

    if "movie" in globals() and str(movie).strip():
        selected_name = str(movie).strip()

        selected_match = titles[
            titles["title_name"].astype(str).str.strip().str.lower()
            == selected_name.lower()
        ]

        if selected_match.empty:
            selected_match = titles[
                titles["title_name"].astype(str).str.contains(
                    selected_name, case=False, na=False
                )
            ]

        if not selected_match.empty:
            selected_chart_title_id = selected_match.iloc[0]["title_id"]
            selected_chart_title = str(selected_match.iloc[0]["title_name"])

            selected_chart_watch = merged[
                merged["title_id"] == selected_chart_title_id
            ]

            if len(selected_chart_watch):
                selected_watch_time = selected_chart_watch[
                    "watch_duration_min"
                ].sum()

                # Keep exactly 8 bars. If selected movie is already in
                # top 8, simply move it to the top. Otherwise replace
                # the lowest-ranked title with the selected movie.
                if selected_chart_title in platform_top.index:
                    top_titles_dash = platform_top.drop(
                        selected_chart_title
                    )
                    top_titles_dash = pd.concat([
                        top_titles_dash,
                        pd.Series({
                            selected_chart_title: selected_watch_time
                        })
                    ])
                else:
                    top_titles_dash = platform_top.iloc[:-1].copy()
                    top_titles_dash.loc[selected_chart_title] = selected_watch_time
            else:
                top_titles_dash = platform_top.copy()
        else:
            top_titles_dash = platform_top.copy()
    else:
        top_titles_dash = platform_top.copy()

except Exception:
    top_titles_dash = pd.Series(dtype=float)
    selected_chart_title = None

try:
    genre_dash = titles["primary_genre"].value_counts().head(8)
except Exception:
    genre_dash = pd.Series(dtype=float)

try:
    country_dash = subscribers["country"].value_counts().head(8)
except Exception:
    country_dash = pd.Series(dtype=float)

try:
    region_dash = subscribers["region"].value_counts().head(8)
except Exception:
    region_dash = pd.Series(dtype=float)

try:
    device_dash = merged["device"].value_counts()
except Exception:
    device_dash = pd.Series(dtype=float)

try:
    language_dash = merged["language"].value_counts().head(8)
except Exception:
    language_dash = pd.Series(dtype=float)

try:
    plan_dash = subscribers["plan_type"].value_counts()
except Exception:
    plan_dash = pd.Series(dtype=float)

try:
    sentiment_dash = reviews["sentiment"].astype(str).str.title().value_counts()
except Exception:
    sentiment_dash = pd.Series(dtype=float)

try:
    corr_dash = watch_history.select_dtypes(include=np.number).corr()
except Exception:
    corr_dash = pd.DataFrame()

try:
    rating_actual = np.asarray(y_test).reshape(-1)
    rating_pred = np.asarray(y_pred).reshape(-1)
    n = min(len(rating_actual), len(rating_pred))
    rating_actual = rating_actual[:n]
    rating_pred = rating_pred[:n]
except Exception:
    rating_actual = np.array([])
    rating_pred = np.array([])

try:
    cm_dash = confusion_matrix(y_test, y_pred)
except Exception:
    cm_dash = np.zeros((2, 2), dtype=int)

# ==========================================================
# DASHBOARD 1
# EXECUTIVE BUSINESS OVERVIEW
# ==========================================================

fig1 = plt.figure(figsize=(20, 12), facecolor=BG)

header(
    fig1,
    "STREAMFLIX EXECUTIVE OVERVIEW",
    "Business performance • audience engagement • platform health",
    1,
    CYAN
)

# KPI ROW
kpi(fig1, 0.055, f"{subscribers['subscriber_id'].nunique():,}", "TOTAL USERS", CYAN)
kpi(fig1, 0.225, f"{titles['title_id'].nunique():,}", "TOTAL TITLES", GREEN)
kpi(fig1, 0.395, f"{watch_history['watch_duration_min'].sum()/60:,.0f}", "WATCH HOURS", ORANGE)
kpi(fig1, 0.565, f"{df['rating'].mean():.2f}", "AVG RATING", PURPLE)
kpi(fig1, 0.735, f"{len(reviews):,}", "TOTAL REVIEWS", PINK)

# Charts row 1
ax = fig1.add_axes([0.055, 0.585, 0.275, 0.190])
chart_title(ax, "Monthly Watch Trend", "Total hours watched", CYAN)

if len(monthly_dash):
    x = np.arange(len(monthly_dash))
    y = monthly_dash.values
    ax.plot(x, y, color=CYAN, linewidth=2.5, marker="o", markersize=3)
    ax.fill_between(x, y, color=CYAN, alpha=.08)

    if len(monthly_dash) <= 15:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(v) for v in monthly_dash.index],
            rotation=30,
            ha="right",
            fontsize=7
        )

ax.set_ylabel("Hours", fontsize=7)

ax = fig1.add_axes([0.365, 0.585, 0.275, 0.190])

chart_title(
    ax,
    "Top Watched Titles",
    "Highest viewing time • selected movie moves to top",
    PURPLE
)

if len(top_titles_dash):
    # Keep the normal Top Watched Titles ranking, but ALWAYS place
    # the currently selected movie at the very top of the chart.
    if selected_chart_title is not None and selected_chart_title in top_titles_dash.index:
        other_titles = top_titles_dash.drop(selected_chart_title).sort_values()
        selected_value = top_titles_dash.loc[selected_chart_title]
        s = pd.concat([
            other_titles,
            pd.Series({selected_chart_title: selected_value})
        ])
    else:
        s = top_titles_dash.sort_values()

    ax.barh(
        np.arange(len(s)),
        s.values,
        color=PURPLE,
        height=.58
    )
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(short(s.index, 22), fontsize=7)

    # Highlight only the selected movie's bar while keeping the chart design same.
    if selected_chart_title is not None and selected_chart_title in s.index:
        selected_pos = list(s.index).index(selected_chart_title)
        ax.patches[selected_pos].set_color(GOLD)

ax.set_xlabel("Minutes", fontsize=7)

ax = fig1.add_axes([0.675, 0.585, 0.270, 0.190])
chart_title(ax, "Genre Portfolio", "Titles by genre", ORANGE)

if len(genre_dash):
    s = genre_dash.sort_values()
    ax.barh(
        np.arange(len(s)),
        s.values,
        color=ORANGE,
        height=.58
    )
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(short(s.index, 18), fontsize=7)

ax.set_xlabel("Titles", fontsize=7)

# Charts row 2
ax = fig1.add_axes([0.055, 0.310, 0.275, 0.190])
chart_title(ax, "Device Mix", "Where users watch", GREEN)

if len(device_dash):
    ax.pie(
        device_dash.values,
        labels=device_dash.index.astype(str),
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=.78,
        wedgeprops={
            "width": .38,
            "edgecolor": BG,
            "linewidth": 2
        },
        textprops={
            "color": WHITE,
            "fontsize": 7
        }
    )
ax.set_aspect("equal")

ax = fig1.add_axes([0.365, 0.310, 0.275, 0.190])
chart_title(ax, "Subscriber Countries", "Largest user markets", BLUE)

if len(country_dash):
    s = country_dash.sort_values()
    ax.barh(
        np.arange(len(s)),
        s.values,
        color=BLUE,
        height=.58
    )
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(short(s.index, 18), fontsize=7)

ax.set_xlabel("Subscribers", fontsize=7)

ax = fig1.add_axes([0.675, 0.310, 0.270, 0.190])
chart_title(ax, "Subscription Plans", "Plan adoption", CYAN)

if len(plan_dash):
    ax.bar(
        np.arange(len(plan_dash)),
        plan_dash.values,
        color=CYAN,
        width=.58
    )
    ax.set_xticks(np.arange(len(plan_dash)))
    ax.set_xticklabels(
        plan_dash.index.astype(str),
        rotation=20,
        ha="right",
        fontsize=7
    )

ax.set_ylabel("Users", fontsize=7)

footer(
    fig1,
    "STREAMFLIX • EXECUTIVE BUSINESS INTELLIGENCE",
    CYAN
)

plt.show(block=True)
plt.close(fig1)

# ==========================================================
# DASHBOARD 2
# CONTENT & USER INTELLIGENCE
# ==========================================================

fig2 = plt.figure(figsize=(20, 12), facecolor=BG)

header(
    fig2,
    "STREAMFLIX CONTENT & USER INTELLIGENCE",
    "Content portfolio • customer behavior • audience segmentation",
    2,
    PURPLE
)

# KPI row
active = int(subscribers["is_active"].sum())
inactive = int((~subscribers["is_active"]).sum())

kpi(fig2, 0.055, f"{active:,}", "ACTIVE USERS", GREEN)
kpi(fig2, 0.225, f"{inactive:,}", "INACTIVE USERS", RED)
kpi(fig2, 0.395, f"{len(watchlist):,}", "WATCHLIST ITEMS", BLUE)
kpi(fig2, 0.565, f"{len(reviews):,}", "REVIEWS", PINK)
kpi(fig2, 0.735, f"{len(watch_history):,}", "WATCH SESSIONS", ORANGE)

# Row 1
ax = fig2.add_axes([0.055, 0.585, 0.275, 0.190])
chart_title(ax, "Regional Audience", "Subscriber concentration", PURPLE)

if len(region_dash):
    s = region_dash.sort_values()
    ax.barh(
        np.arange(len(s)),
        s.values,
        color=PURPLE,
        height=.58
    )
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(short(s.index, 18), fontsize=7)

ax.set_xlabel("Subscribers", fontsize=7)

ax = fig2.add_axes([0.365, 0.585, 0.275, 0.190])
chart_title(ax, "Language Landscape", "Audience language distribution", CYAN)

if len(language_dash):
    s = language_dash.sort_values()
    ax.barh(
        np.arange(len(s)),
        s.values,
        color=CYAN,
        height=.58
    )
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(short(s.index, 16), fontsize=7)

ax.set_xlabel("Users", fontsize=7)

ax = fig2.add_axes([0.675, 0.585, 0.270, 0.190])
chart_title(ax, "Review Sentiment", "Voice of the customer", PINK)

if len(sentiment_dash):
    colors = []
    for v in sentiment_dash.index:
        name = str(v).lower()
        if "positive" in name:
            colors.append(GREEN)
        elif "negative" in name:
            colors.append(RED)
        else:
            colors.append(GOLD)

    ax.bar(
        np.arange(len(sentiment_dash)),
        sentiment_dash.values,
        color=colors,
        width=.58
    )
    ax.set_xticks(np.arange(len(sentiment_dash)))
    ax.set_xticklabels(
        sentiment_dash.index,
        rotation=15,
        ha="right",
        fontsize=7
    )

ax.set_ylabel("Reviews", fontsize=7)

# Row 2
ax = fig2.add_axes([0.055, 0.310, 0.275, 0.190])
chart_title(ax, "Watch Duration Distribution", "Session behavior", CYAN)

try:
    h = watch_history["watch_duration_min"].dropna()
    upper = h.quantile(.99)
    h = h[h <= upper]

    ax.hist(
        h,
        bins=28,
        color=CYAN,
        alpha=.75,
        edgecolor=BG
    )
except Exception:
    pass

ax.set_xlabel("Minutes", fontsize=7)
ax.set_ylabel("Sessions", fontsize=7)

ax = fig2.add_axes([0.365, 0.310, 0.275, 0.190])
chart_title(ax, "Watch Duration Outliers", "Behavioral spread", PINK)

try:
    d = watch_history["watch_duration_min"].dropna()

    ax.boxplot(
        d,
        patch_artist=True,
        boxprops=dict(
            facecolor="#233B5D",
            edgecolor=PINK,
            linewidth=1.2
        ),
        medianprops=dict(
            color=WHITE,
            linewidth=2
        ),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(
            marker="o",
            markersize=2.5,
            markerfacecolor=PINK,
            markeredgecolor=PINK,
            alpha=.4
        )
    )
except Exception:
    pass

ax.set_ylabel("Minutes", fontsize=7)

ax = fig2.add_axes([0.675, 0.310, 0.270, 0.190])
chart_title(ax, "Content Ratings", "Rating distribution", GOLD)

try:
    ratings_dash = df["rating"].dropna()

    bins = np.arange(
        ratings_dash.min() - .25,
        ratings_dash.max() + .26,
        .5
    )

    ax.hist(
        ratings_dash,
        bins=bins,
        color=GOLD,
        alpha=.78,
        edgecolor=BG
    )
except Exception:
    pass

ax.set_xlabel("Rating", fontsize=7)
ax.set_ylabel("Count", fontsize=7)

footer(
    fig2,
    "STREAMFLIX • CONTENT STRATEGY & USER INTELLIGENCE",
    PURPLE
)

plt.show(block=True)
plt.close(fig2)

# ==========================================================
# DASHBOARD 3
# AI & MACHINE LEARNING LAB
# ==========================================================

fig3 = plt.figure(figsize=(20, 12), facecolor=BG)

header(
    fig3,
    "STREAMFLIX AI & MACHINE LEARNING LAB",
    "Prediction performance • model diagnostics • intelligent insights",
    3,
    CYAN
)

# ML KPI
try:
    model_acc = float(acc) * 100
except Exception:
    model_acc = 0

try:
    model_r2 = float(r2_score(y_test, y_pred))
except Exception:
    model_r2 = 0

try:
    model_mae = float(mean_absolute_error(y_test, y_pred))
except Exception:
    model_mae = 0

try:
    model_rmse = float(mean_squared_error(y_test, y_pred) ** .5)
except Exception:
    model_rmse = 0

kpi(fig3, 0.055, f"{model_acc:.2f}%", "MODEL ACCURACY", GREEN)
kpi(fig3, 0.225, f"{model_r2:.2f}", "R² SCORE", CYAN)
kpi(fig3, 0.395, f"{model_mae:.2f}", "MAE", ORANGE)
kpi(fig3, 0.565, f"{model_rmse:.2f}", "RMSE", PINK)
kpi(fig3, 0.735, f"{len(y_test):,}", "TEST RECORDS", PURPLE)

# Row 1
ax = fig3.add_axes([0.055, 0.585, 0.275, 0.190])
chart_title(ax, "Feature Importance", "Random Forest drivers", PURPLE)

try:
    fi = importance.head(8).sort_values("Importance")

    ax.barh(
        np.arange(len(fi)),
        fi["Importance"].values,
        color=PURPLE,
        height=.58
    )

    ax.set_yticks(np.arange(len(fi)))
    ax.set_yticklabels(
        short(fi["Feature"], 18),
        fontsize=7
    )
except Exception:
    pass

ax.set_xlabel("Importance", fontsize=7)

ax = fig3.add_axes([0.365, 0.585, 0.275, 0.190])
chart_title(ax, "Actual vs Predicted", "Prediction quality", CYAN)

if len(rating_actual):
    mask = (
        np.isfinite(rating_actual) &
        np.isfinite(rating_pred)
    )

    a = rating_actual[mask]
    p = rating_pred[mask]

    ax.scatter(
        a,
        p,
        s=14,
        alpha=.35,
        color=CYAN,
        edgecolors="none"
    )

    lo = min(a.min(), p.min())
    hi = max(a.max(), p.max())

    ax.plot(
        [lo, hi],
        [lo, hi],
        color=PINK,
        linewidth=2,
        linestyle="--"
    )

ax.set_xlabel("Actual", fontsize=7)
ax.set_ylabel("Predicted", fontsize=7)

ax = fig3.add_axes([0.675, 0.585, 0.270, 0.190])
chart_title(ax, "Residual Diagnostics", "Error around zero", ORANGE)

try:
    pred_arr = np.asarray(y_pred).reshape(-1)
    actual_arr = np.asarray(y_test).reshape(-1)

    n = min(len(pred_arr), len(actual_arr))

    pred_arr = pred_arr[:n]
    actual_arr = actual_arr[:n]

    res = actual_arr - pred_arr

    mask = (
        np.isfinite(pred_arr) &
        np.isfinite(res)
    )

    ax.scatter(
        pred_arr[mask],
        res[mask],
        s=13,
        alpha=.35,
        color=ORANGE,
        edgecolors="none"
    )

    ax.axhline(
        0,
        color=PINK,
        linewidth=2,
        linestyle="--"
    )
except Exception:
    pass

ax.set_xlabel("Predicted", fontsize=7)
ax.set_ylabel("Residual", fontsize=7)

# Row 2
ax = fig3.add_axes([0.055, 0.310, 0.275, 0.190])
chart_title(ax, "Confusion Matrix", "Churn classification", GREEN)

try:
    im = ax.imshow(
        cm_dash,
        cmap="magma",
        aspect="auto"
    )

    for i in range(cm_dash.shape[0]):
        for j in range(cm_dash.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_dash[i, j]:,}",
                ha="center",
                va="center",
                color=WHITE,
                fontsize=11,
                fontweight="bold"
            )

    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)

    if cm_dash.shape == (2, 2):
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(
            ["Non-Churn", "Churn"],
            fontsize=7
        )
        ax.set_yticklabels(
            ["Non-Churn", "Churn"],
            fontsize=7
        )
except Exception:
    pass

ax = fig3.add_axes([0.365, 0.310, 0.275, 0.190])
chart_title(ax, "Correlation Matrix", "Numeric relationships", PURPLE)

try:
    if not corr_dash.empty:
        cols = list(corr_dash.columns[:8])
        c = corr_dash.loc[cols, cols]

        ax.imshow(
            c,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))

        ax.set_xticklabels(
            short(cols, 12),
            rotation=55,
            ha="right",
            fontsize=6
        )

        ax.set_yticklabels(
            short(cols, 12),
            fontsize=6
        )
except Exception:
    pass

ax = fig3.add_axes([0.675, 0.310, 0.270, 0.190])
chart_title(ax, "Prediction Error Distribution", "Model residual frequency", PINK)

try:
    if len(rating_actual):
        residuals = rating_actual - rating_pred

        ax.hist(
            residuals,
            bins=28,
            color=PINK,
            alpha=.72,
            edgecolor=BG
        )

        ax.axvline(
            0,
            color=WHITE,
            linewidth=1.5,
            linestyle="--"
        )
except Exception:
    pass

ax.set_xlabel("Residual", fontsize=7)
ax.set_ylabel("Frequency", fontsize=7)

footer(
    fig3,
    "STREAMFLIX • AI / ML LAB • SCIKIT-LEARN • PANDAS • MATPLOTLIB",
    CYAN
)

plt.show(block=True)
plt.close(fig3)

print("\n" + "=" * 80)
print("STREAMFLIX — 3 PROFESSIONAL DASHBOARDS COMPLETED")
print("=" * 80)
print("1. Executive Business Overview")
print("2. Content & User Intelligence")
print("3. AI & Machine Learning Lab")
print("Each dashboard is grouped into ONE clean window.")
print("No standalone chart windows.")
print("=" * 80)


# ==========================================================
# 🎬 SELECTED MOVIE ANALYTICS DASHBOARD
# ==========================================================
# The movie entered/selected above is automatically analyzed.
# All charts below are filtered only for that movie.
# ==========================================================

try:
    selected_movie_name = str(movie).strip()

    # Find selected title (case-insensitive)
    title_match = titles[
        titles["title_name"].astype(str).str.strip().str.lower()
        == selected_movie_name.lower()
    ]

    # If exact match is not found, try partial match
    if title_match.empty:
        title_match = titles[
            titles["title_name"].astype(str).str.contains(
                selected_movie_name,
                case=False,
                na=False
            )
        ]

    if not title_match.empty:

        selected_row = title_match.iloc[0]
        selected_title_id = selected_row["title_id"]
        selected_movie_name = str(selected_row["title_name"])

        # --------------------------------------------------
        # FILTER MOVIE DATA
        # --------------------------------------------------
        movie_watch = watch_history[
            watch_history["title_id"] == selected_title_id
        ].copy()

        movie_ratings = df[
            df["title_id"] == selected_title_id
        ].copy()

        # watchlist may use title_id
        if "title_id" in watchlist.columns:
            movie_watchlist = watchlist[
                watchlist["title_id"] == selected_title_id
            ].copy()
        else:
            movie_watchlist = pd.DataFrame()

        # reviews normally use movie_id in this project
        if "movie_id" in reviews.columns:
            movie_reviews = reviews[
                reviews["movie_id"] == selected_title_id
            ].copy()
        elif "title_id" in reviews.columns:
            movie_reviews = reviews[
                reviews["title_id"] == selected_title_id
            ].copy()
        else:
            movie_reviews = pd.DataFrame()

        # --------------------------------------------------
        # SAFE KPI CALCULATIONS
        # --------------------------------------------------
        total_views = len(movie_watch)

        total_watch_hours = (
            movie_watch["watch_duration_min"].sum() / 60
            if "watch_duration_min" in movie_watch.columns else 0
        )

        avg_watch_duration = (
            movie_watch["watch_duration_min"].mean()
            if len(movie_watch) and "watch_duration_min" in movie_watch.columns
            else 0
        )

        avg_completion = (
            movie_watch["completion_pct"].mean()
            if len(movie_watch) and "completion_pct" in movie_watch.columns
            else 0
        )

        if "completed" in movie_watch.columns and len(movie_watch):
            completed_views = int(movie_watch["completed"].astype(bool).sum())
        else:
            completed_views = 0

        incomplete_views = max(total_views - completed_views, 0)

        completion_rate = (
            completed_views / total_views * 100
            if total_views else 0
        )

        avg_rating = (
            movie_ratings["rating"].mean()
            if len(movie_ratings) and "rating" in movie_ratings.columns
            else 0
        )

        total_ratings = len(movie_ratings)
        total_reviews = len(movie_reviews)
        total_watchlist = len(movie_watchlist)

        # --------------------------------------------------
        # CHART DATA
        # --------------------------------------------------
        if "device" in movie_watch.columns:
            device_data = movie_watch["device"].value_counts().head(8)
        else:
            device_data = pd.Series(dtype=float)

        if "region" in movie_watch.columns:
            region_data = movie_watch["region"].value_counts().head(8)
        else:
            region_data = pd.Series(dtype=float)

        if "rating" in movie_ratings.columns:
            rating_data = movie_ratings["rating"].dropna().value_counts().sort_index()
        else:
            rating_data = pd.Series(dtype=float)

        if "watch_date" in movie_watch.columns:
            movie_watch["watch_date"] = pd.to_datetime(
                movie_watch["watch_date"],
                errors="coerce"
            )

            daily_views = (
                movie_watch.dropna(subset=["watch_date"])
                .groupby(movie_watch.dropna(subset=["watch_date"])["watch_date"].dt.date)
                .size()
            )

            daily_hours = (
                movie_watch.dropna(subset=["watch_date"])
                .groupby(movie_watch.dropna(subset=["watch_date"])["watch_date"].dt.date)
                ["watch_duration_min"].sum() / 60
                if "watch_duration_min" in movie_watch.columns
                else pd.Series(dtype=float)
            )
        else:
            daily_views = pd.Series(dtype=float)
            daily_hours = pd.Series(dtype=float)

        if "watch_duration_min" in movie_watch.columns:
            duration_data = movie_watch["watch_duration_min"].dropna()
        else:
            duration_data = pd.Series(dtype=float)

        if "sentiment" in movie_reviews.columns:
            sentiment_data = movie_reviews["sentiment"].astype(str).str.title().value_counts()
        else:
            sentiment_data = pd.Series(dtype=float)

        # --------------------------------------------------
        # MOVIE DASHBOARD
        # --------------------------------------------------
        fig_movie = plt.figure(
            figsize=(20, 12),
            facecolor=BG
        )

        header(
            fig_movie,
            f"STREAMFLIX • {selected_movie_name.upper()}",
            "Selected movie performance • audience behavior • viewing intelligence",
            4,
            GOLD
        )

        # KPI ROW
        kpi(fig_movie, 0.055, f"{total_views:,}", "TOTAL VIEWS", CYAN)
        kpi(fig_movie, 0.225, f"{total_watch_hours:,.1f}", "WATCH HOURS", ORANGE)
        kpi(fig_movie, 0.395, f"{avg_rating:.2f}", "AVG RATING", GOLD)
        kpi(fig_movie, 0.565, f"{avg_completion:.1f}%", "AVG COMPLETION", GREEN)
        kpi(fig_movie, 0.735, f"{total_reviews:,}", "REVIEWS", PINK)

        # --------------------------------------------------
        # 1. VIEWING TREND
        # --------------------------------------------------
        ax = fig_movie.add_axes([0.055, 0.585, 0.275, 0.190])
        chart_title(
            ax,
            "Viewing Trend",
            "Daily views for selected movie",
            CYAN
        )

        if len(daily_views):
            x = np.arange(len(daily_views))
            y = daily_views.values
            ax.plot(
                x, y,
                color=CYAN,
                linewidth=2.5,
                marker="o",
                markersize=3
            )
            ax.fill_between(x, y, color=CYAN, alpha=.08)

            if len(daily_views) <= 15:
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [str(v) for v in daily_views.index],
                    rotation=35,
                    ha="right",
                    fontsize=6
                )

        ax.set_ylabel("Views", fontsize=7)

        # --------------------------------------------------
        # 2. DEVICE USAGE
        # --------------------------------------------------
        ax = fig_movie.add_axes([0.365, 0.585, 0.275, 0.190])
        chart_title(
            ax,
            "Device Usage",
            "How viewers watched this movie",
            GREEN
        )

        if len(device_data):
            s = device_data.sort_values()
            ax.barh(
                np.arange(len(s)),
                s.values,
                color=GREEN,
                height=.58
            )
            ax.set_yticks(np.arange(len(s)))
            ax.set_yticklabels(short(s.index, 18), fontsize=7)

        ax.set_xlabel("Views", fontsize=7)

        # --------------------------------------------------
        # 3. COMPLETION ANALYSIS
        # --------------------------------------------------
        ax = fig_movie.add_axes([0.675, 0.585, 0.270, 0.190])
        chart_title(
            ax,
            "Completion Analysis",
            "Completed vs incomplete views",
            PURPLE
        )

        completion_data = pd.Series({
            "Completed": completed_views,
            "Incomplete": incomplete_views
        })

        if completion_data.sum() > 0:
            ax.pie(
                completion_data.values,
                labels=completion_data.index,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=.78,
                wedgeprops={
                    "width": .38,
                    "edgecolor": BG,
                    "linewidth": 2
                },
                textprops={
                    "color": WHITE,
                    "fontsize": 7
                }
            )

        ax.set_aspect("equal")

        # --------------------------------------------------
        # 4. RATING DISTRIBUTION
        # --------------------------------------------------
        ax = fig_movie.add_axes([0.055, 0.310, 0.275, 0.190])
        chart_title(
            ax,
            "Rating Distribution",
            "Audience ratings for this movie",
            GOLD
        )

        if len(rating_data):
            x = np.arange(len(rating_data))
            ax.bar(
                x,
                rating_data.values,
                color=GOLD,
                width=.58
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                rating_data.index,
                fontsize=7
            )

        ax.set_xlabel("Rating", fontsize=7)
        ax.set_ylabel("Users", fontsize=7)

        # --------------------------------------------------
        # 5. WATCH HOURS TREND
        # --------------------------------------------------
        ax = fig_movie.add_axes([0.365, 0.310, 0.275, 0.190])
        chart_title(
            ax,
            "Watch Hours Trend",
            "Viewing hours over time",
            ORANGE
        )

        if len(daily_hours):
            x = np.arange(len(daily_hours))
            y = daily_hours.values
            ax.plot(
                x, y,
                color=ORANGE,
                linewidth=2.5,
                marker="o",
                markersize=3
            )
            ax.fill_between(x, y, color=ORANGE, alpha=.08)

            if len(daily_hours) <= 15:
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [str(v) for v in daily_hours.index],
                    rotation=35,
                    ha="right",
                    fontsize=6
                )

        ax.set_ylabel("Hours", fontsize=7)

        # --------------------------------------------------
        # 6. REVIEW SENTIMENT / WATCH DURATION
        # --------------------------------------------------
        ax = fig_movie.add_axes([0.675, 0.310, 0.270, 0.190])

        if len(sentiment_data):
            chart_title(
                ax,
                "Review Sentiment",
                "Audience feedback for this movie",
                PINK
            )
            s = sentiment_data.sort_values()
            ax.barh(
                np.arange(len(s)),
                s.values,
                color=PINK,
                height=.58
            )
            ax.set_yticks(np.arange(len(s)))
            ax.set_yticklabels(s.index, fontsize=7)
            ax.set_xlabel("Reviews", fontsize=7)
        else:
            chart_title(
                ax,
                "Watch Duration",
                "Distribution of viewing duration",
                PINK
            )
            if len(duration_data):
                ax.hist(
                    duration_data,
                    bins=15,
                    color=PINK,
                    alpha=.78,
                    edgecolor=BG
                )
            ax.set_xlabel("Minutes", fontsize=7)
            ax.set_ylabel("Views", fontsize=7)

        # --------------------------------------------------
        # MOVIE INFO FOOTER
        # --------------------------------------------------
        footer(
            fig_movie,
            f"STREAMFLIX • SELECTED MOVIE ANALYTICS • {selected_movie_name.upper()} • "
            f"Views: {total_views:,} • Rating: {avg_rating:.2f} • Completion: {completion_rate:.1f}%",
            GOLD
        )

        plt.show(block=True)
        plt.close(fig_movie)

        print("\n" + "="*70)
        print("SELECTED MOVIE ANALYSIS COMPLETED")
        print("Movie:", selected_movie_name)
        print("="*70)

    else:
        print("\n❌ Selected movie was not found in titles.csv")
        print("Please enter the exact movie name shown in Available Movies.")

except Exception as movie_dashboard_error:
    print("\n❌ Movie Dashboard Error:", movie_dashboard_error)
    print("The main StreamFlix dashboards can still run normally.")


# ============================================================================
# STREAMFLIX PROJECT — ASSIGNMENT REQUIRED ADDITIONS / FINAL UPDATE
# ============================================================================
# This section was added to the original project so that the required items
# from the StreamFlix PRD are also covered, without removing the original
# analysis, ML models, recommendation system, or dashboards.
#
# REQUIRED PHASE 1: Data Quality / Cleaning checks
# REQUIRED PHASE 2: All 10 specified EDA charts + observations
# REQUIRED PHASE 3: All 10 specified business KPIs
# EXTRA: Static 5-page dashboard-style outputs + management summary
# ============================================================================

from pathlib import Path as _Path
import math as _math

_REQUIRED_OUT = _Path(__file__).resolve().parent / "streamflix_required_outputs"
_REQUIRED_OUT.mkdir(exist_ok=True)

print("\n" + "=" * 90)
print("STREAMFLIX — ASSIGNMENT REQUIRED SECTION")
print("=" * 90)

# ---------------------------------------------------------------------------
# A. PREPARE ANALYSIS COPIES
# ---------------------------------------------------------------------------
_req_sub = subscribers.copy()
_req_titles = titles.copy()
_req_watch = watch_history.copy()
_req_ratings = df.copy()
_req_reviews = reviews.copy()
_req_watchlist = watchlist.copy()

_req_sub["signup_date"] = pd.to_datetime(_req_sub["signup_date"], errors="coerce")
_req_sub["churn_date"] = pd.to_datetime(_req_sub["churn_date"], errors="coerce")
_req_titles["date_added"] = pd.to_datetime(_req_titles["date_added"], errors="coerce")
_req_titles["license_expiry"] = pd.to_datetime(_req_titles["license_expiry"], errors="coerce")
_req_watch["watch_date"] = pd.to_datetime(_req_watch["watch_date"], errors="coerce")
_req_ratings["rating_date"] = pd.to_datetime(_req_ratings["rating_date"], errors="coerce")
_req_reviews["review_date"] = pd.to_datetime(_req_reviews["review_date"], errors="coerce")
_req_watchlist["added_date"] = pd.to_datetime(_req_watchlist["added_date"], errors="coerce")

# ---------------------------------------------------------------------------
# PHASE 1 — DATA QUALITY / CLEANING REPORT
# ---------------------------------------------------------------------------
print("\n" + "-" * 90)
print("PHASE 1 — DATA QUALITY / CLEANING CHECKS")
print("-" * 90)

_tables = {
    "subscribers": _req_sub,
    "titles": _req_titles,
    "watch_history": _req_watch,
    "ratings": _req_ratings,
    "reviews": _req_reviews,
    "watchlist": _req_watchlist,
}

_qc_lines = []
_qc_lines.append("STREAMFLIX — PHASE 1 DATA QUALITY REPORT")
_qc_lines.append("=" * 70)
_qc_lines.append("")
_qc_lines.append("TABLE PROFILE")
_qc_lines.append("-" * 70)

for _name, _table in _tables.items():
    _qc_lines.append(
        f"{_name}: rows={len(_table):,}, columns={_table.shape[1]}, "
        f"null_cells={int(_table.isna().sum().sum()):,}, "
        f"duplicate_rows={int(_table.duplicated().sum()):,}"
    )

# Primary-key duplicate checks
_pk_checks = {
    "subscribers.subscriber_id": (_req_sub, "subscriber_id"),
    "titles.title_id": (_req_titles, "title_id"),
    "watch_history.watch_id": (_req_watch, "watch_id"),
    "ratings.rating_id": (_req_ratings, "rating_id"),
    "reviews.review_id": (_req_reviews, "review_id"),
    "watchlist.watchlist_id": (_req_watchlist, "watchlist_id"),
}

_qc_lines.append("")
_qc_lines.append("PRIMARY KEY DUPLICATE CHECKS")
_qc_lines.append("-" * 70)
for _label, (_table, _col) in _pk_checks.items():
    _dup = int(_table[_col].duplicated().sum())
    _qc_lines.append(f"{_label}: {_dup} duplicate IDs")

# Referential integrity
_sub_ids = set(_req_sub["subscriber_id"].dropna())
_title_ids = set(_req_titles["title_id"].dropna())
_watch_bad_sub = int((~_req_watch["subscriber_id"].isin(_sub_ids)).sum())
_watch_bad_title = int((~_req_watch["title_id"].isin(_title_ids)).sum())
_wl_bad_sub = int((~_req_watchlist["subscriber_id"].isin(_sub_ids)).sum())
_wl_bad_title = int((~_req_watchlist["title_id"].isin(_title_ids)).sum())

_qc_lines.append("")
_qc_lines.append("REFERENTIAL INTEGRITY")
_qc_lines.append("-" * 70)
_qc_lines.append(f"watch_history -> subscribers: {_watch_bad_sub} invalid subscriber_id")
_qc_lines.append(f"watch_history -> titles: {_watch_bad_title} invalid title_id")
_qc_lines.append(f"watchlist -> subscribers: {_wl_bad_sub} invalid subscriber_id")
_qc_lines.append(f"watchlist -> titles: {_wl_bad_title} invalid title_id")

# Subscriber date consistency
_bad_churn_order = int(
    ((~_req_sub["is_active"]) &
     _req_sub["churn_date"].notna() &
     (_req_sub["churn_date"] <= _req_sub["signup_date"])).sum()
)
_active_with_churn = int((_req_sub["is_active"] & _req_sub["churn_date"].notna()).sum())
_short_active = int((_req_sub["is_active"] & (_req_sub["tenure_months"] < 3)).sum())

_qc_lines.append("")
_qc_lines.append("SUBSCRIBER DATE / TENURE CHECKS")
_qc_lines.append("-" * 70)
_qc_lines.append(f"Churned rows with churn_date <= signup_date: {_bad_churn_order}")
_qc_lines.append(f"Active subscribers with non-blank churn_date: {_active_with_churn}")
_qc_lines.append(f"Active subscribers with tenure < 3 months (not an error): {_short_active}")

# Watch-history mathematical consistency
_calc_completion = (
    _req_watch["watch_duration_min"] /
    _req_watch["content_duration_min"] * 100
)
_duration_over_content = int(
    (_req_watch["watch_duration_min"] > _req_watch["content_duration_min"]).sum()
)
_completion_mismatch = int(((_req_watch["completion_pct"] - _calc_completion).abs() > 2).sum())
_max_completion_diff = float((_req_watch["completion_pct"] - _calc_completion).abs().max())

_qc_lines.append("")
_qc_lines.append("WATCH HISTORY CONSISTENCY")
_qc_lines.append("-" * 70)
_qc_lines.append(f"Sessions where watch_duration_min > content_duration_min: {_duration_over_content}")
_qc_lines.append(f"completion_pct mismatch > 2 percentage points: {_completion_mismatch}")
_qc_lines.append(f"Maximum completion_pct calculation difference: {_max_completion_diff:.4f}")

# Sentiment validation
_valid_sentiments = {"Positive", "Neutral", "Negative"}
_actual_sentiments = set(_req_reviews["sentiment"].dropna().astype(str).str.title())
_invalid_sentiments = sorted(_actual_sentiments - _valid_sentiments)

_qc_lines.append("")
_qc_lines.append("REVIEW SENTIMENT VALIDATION")
_qc_lines.append("-" * 70)
_qc_lines.append(f"Valid sentiment values: {sorted(_actual_sentiments)}")
_qc_lines.append(f"Invalid sentiment values: {_invalid_sentiments if _invalid_sentiments else 'None'}")

# Date / numeric validity checks
_qc_lines.append("")
_qc_lines.append("DATA TYPE / RANGE CHECKS")
_qc_lines.append("-" * 70)
_qc_lines.append(f"Invalid watch_date values after parsing: {int(_req_watch['watch_date'].isna().sum())}")
_qc_lines.append(f"Invalid signup_date values after parsing: {int(_req_sub['signup_date'].isna().sum())}")
_qc_lines.append(f"Ratings outside 1–5: {int((~_req_ratings['rating'].between(1, 5)).sum())}")
_qc_lines.append(f"completion_pct outside 0–100: {int((~_req_watch['completion_pct'].between(0, 100)).sum())}")

# Overall QC conclusion
_qc_ok = (
    sum(int(_table.duplicated().sum()) for _table in _tables.values()) == 0 and
    all(int(_table[_col].duplicated().sum()) == 0 for _table, _col in _pk_checks.values()) and
    _watch_bad_sub == 0 and _watch_bad_title == 0 and
    _wl_bad_sub == 0 and _wl_bad_title == 0 and
    _bad_churn_order == 0 and _active_with_churn == 0 and
    _duration_over_content == 0 and _completion_mismatch == 0 and
    not _invalid_sentiments
)
_qc_lines.append("")
_qc_lines.append("OVERALL CONCLUSION")
_qc_lines.append("-" * 70)
_qc_lines.append(
    "The core integrity checks passed. Expected blank fields (such as churn_date "
    "for active subscribers or optional title metadata) are treated as missing/blank "
    "rather than automatically as data errors."
    if _qc_ok else
    "One or more integrity checks need attention. Review the specific checks above."
)

(_REQUIRED_OUT / "Phase1_DataQuality_Report.txt").write_text(
    "\n".join(_qc_lines), encoding="utf-8"
)
print("\n".join(_qc_lines))

# ---------------------------------------------------------------------------
# PHASE 2 — EXACTLY THE 10 REQUIRED EDA CHARTS
# ---------------------------------------------------------------------------
print("\n" + "-" * 90)
print("PHASE 2 — REQUIRED 10 EDA CHARTS")
print("-" * 90)

_EDA_OUT = _REQUIRED_OUT / "Phase2_EDA_Charts"
_EDA_OUT.mkdir(exist_ok=True)

_obs = []

def _save_chart(_fig, _filename, _title, _observation):
    _fig.tight_layout()
    _fig.savefig(_EDA_OUT / _filename, dpi=160, bbox_inches="tight")
    plt.close(_fig)
    _obs.append(f"{_title}\n{_observation}\n")
    print(f"Saved: {_filename}")
    print(f"Observation: {_observation}\n")

# 1. Monthly viewing volume
_monthly_sessions = _req_watch.groupby(
    _req_watch["watch_date"].dt.to_period("M")
).size().sort_index()
_fig = plt.figure(figsize=(12, 5))
plt.plot(_monthly_sessions.index.astype(str), _monthly_sessions.values, marker="o")
plt.title("1. Monthly Viewing Volume")
plt.xlabel("Month")
plt.ylabel("Viewing Sessions")
plt.xticks(rotation=45)
plt.grid(True, alpha=.25)
_max_month = str(_monthly_sessions.idxmax())
_max_sessions = int(_monthly_sessions.max())
_save_chart(
    _fig, "01_monthly_viewing_volume.png", "1. Monthly Viewing Volume",
    f"The busiest month was {_max_month} with {_max_sessions:,} viewing sessions. "
    "The line shows how viewing activity changes across the available timeline."
)

# 2. Monthly watch hours trend
_monthly_hours = _req_watch.groupby(
    _req_watch["watch_date"].dt.to_period("M")
)["watch_duration_min"].sum().div(60).sort_index()
_fig = plt.figure(figsize=(12, 5))
plt.plot(_monthly_hours.index.astype(str), _monthly_hours.values, marker="o")
plt.title("2. Monthly Watch Hours Trend")
plt.xlabel("Month")
plt.ylabel("Watch Hours")
plt.xticks(rotation=45)
plt.grid(True, alpha=.25)
_peak_hours_month = str(_monthly_hours.idxmax())
_peak_hours = float(_monthly_hours.max())
_save_chart(
    _fig, "02_monthly_watch_hours.png", "2. Monthly Watch Hours Trend",
    f"Watch time peaked in {_peak_hours_month} at approximately {_peak_hours:,.1f} hours. "
    "This trend is useful for identifying periods of unusually high or low engagement."
)

# Merge watch history with titles once for content-level EDA
_req_wh_titles = _req_watch.merge(
    _req_titles[["title_id", "primary_genre", "type", "country", "language", "is_original", "title_name"]],
    on="title_id", how="left"
)

# 3. Genre-wise watch-hours share
_genre_hours = _req_wh_titles.groupby("primary_genre")["watch_duration_min"].sum().div(60).sort_values(ascending=False)
_fig = plt.figure(figsize=(10, 6))
if len(_genre_hours) > 6:
    _genre_plot = _genre_hours.head(10).sort_values()
    plt.barh(_genre_plot.index.astype(str), _genre_plot.values)
    plt.xlabel("Watch Hours")
else:
    plt.pie(_genre_hours.values, labels=_genre_hours.index, autopct="%1.1f%%", startangle=90)
plt.title("3. Genre-wise Watch-hours Share")
_top_genre = str(_genre_hours.index[0])
_top_genre_hours = float(_genre_hours.iloc[0])
_save_chart(
    _fig, "03_genre_watch_hours_share.png", "3. Genre-wise Watch-hours Share",
    f"{_top_genre} generated the most watch time at approximately {_top_genre_hours:,.1f} hours. "
    "The chart highlights which content categories contribute most to total engagement."
)

# 4. Content type split — Movies vs TV Shows
_type_hours = _req_wh_titles.groupby("type")["watch_duration_min"].sum().div(60)
_fig = plt.figure(figsize=(7, 6))
plt.pie(_type_hours.values, labels=_type_hours.index, autopct="%1.1f%%", startangle=90)
plt.title("4. Content Type Split — Movies vs TV Shows")
_top_type = str(_type_hours.idxmax())
_save_chart(
    _fig, "04_content_type_split.png", "4. Content Type Split",
    f"{_top_type} accounts for the larger share of watch hours. "
    "This helps compare engagement generated by Movies versus TV Shows."
)

# 5. Top 10 countries by watch hours
_country_hours = _req_wh_titles.groupby("country")["watch_duration_min"].sum().div(60).sort_values(ascending=False).head(10).sort_values()
_fig = plt.figure(figsize=(10, 6))
plt.barh(_country_hours.index.astype(str), _country_hours.values)
plt.title("5. Top 10 Countries by Watch Hours")
plt.xlabel("Watch Hours")
_country_top = str(_country_hours.index[-1])
_save_chart(
    _fig, "05_top10_countries_watch_hours.png", "5. Top 10 Countries by Watch Hours",
    f"{_country_top} is the leading country among the top 10 by total watch hours. "
    "Country-level engagement can help guide market and content decisions."
)

# 6. Subscriber plan distribution — subscriber count, not watch hours
_plan_counts = _req_sub["plan_type"].value_counts()
_fig = plt.figure(figsize=(7, 6))
plt.pie(_plan_counts.values, labels=_plan_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("6. Subscriber Plan Distribution")
_top_plan = str(_plan_counts.index[0])
_save_chart(
    _fig, "06_subscriber_plan_distribution.png", "6. Subscriber Plan Distribution",
    f"{_top_plan} has the largest subscriber count. "
    "This chart measures plan adoption by subscribers rather than watch time."
)

# 7. Device usage
_device_counts = _req_watch["device"].value_counts().sort_values(ascending=False)
_fig = plt.figure(figsize=(9, 5))
plt.bar(_device_counts.index.astype(str), _device_counts.values)
plt.title("7. Device Usage")
plt.xlabel("Device")
plt.ylabel("Sessions")
plt.xticks(rotation=30)
plt.grid(True, axis="y", alpha=.25)
_top_device = str(_device_counts.index[0])
_save_chart(
    _fig, "07_device_usage.png", "7. Device Usage",
    f"{_top_device} is the most frequently used device for viewing sessions. "
    "Device mix can help prioritize product and playback experience improvements."
)

# 8. Age distribution of subscribers
_fig = plt.figure(figsize=(9, 5))
plt.hist(_req_sub["age"].dropna(), bins=15, edgecolor="black")
plt.title("8. Age Distribution of Subscribers")
plt.xlabel("Age")
plt.ylabel("Subscribers")
plt.grid(True, axis="y", alpha=.25)
_age_mode_bin = int(_req_sub["age"].mode().iloc[0])
_save_chart(
    _fig, "08_age_distribution.png", "8. Age Distribution of Subscribers",
    f"The most common individual subscriber age is {_age_mode_bin}. "
    "The histogram shows how the subscriber base is distributed across age groups."
)

# 9. Completion rate by genre
_completion_by_genre = _req_wh_titles.groupby("primary_genre")["completion_pct"].mean().sort_values(ascending=False)
_fig = plt.figure(figsize=(10, 6))
_plot_completion = _completion_by_genre.sort_values()
plt.barh(_plot_completion.index.astype(str), _plot_completion.values)
plt.title("9. Completion Rate by Genre")
plt.xlabel("Average Completion %")
plt.xlim(0, 100)
_best_completion_genre = str(_completion_by_genre.index[0])
_best_completion = float(_completion_by_genre.iloc[0])
_save_chart(
    _fig, "09_completion_rate_by_genre.png", "9. Completion Rate by Genre",
    f"{_best_completion_genre} has the highest average completion rate at {_best_completion:.1f}%. "
    "Higher completion indicates that viewers tend to consume more of the content in that genre."
)

# 10. Review sentiment breakdown
_sentiment_counts = _req_reviews["sentiment"].astype(str).str.title().value_counts()
_sentiment_order = [x for x in ["Positive", "Neutral", "Negative"] if x in _sentiment_counts.index]
_sentiment_counts = _sentiment_counts.reindex(_sentiment_order).fillna(0)
_fig = plt.figure(figsize=(8, 5))
plt.bar(_sentiment_counts.index, _sentiment_counts.values)
plt.title("10. Review Sentiment Breakdown")
plt.xlabel("Sentiment")
plt.ylabel("Reviews")
plt.grid(True, axis="y", alpha=.25)
_top_sentiment = str(_sentiment_counts.idxmax())
_top_sentiment_count = int(_sentiment_counts.max())
_save_chart(
    _fig, "10_review_sentiment_breakdown.png", "10. Review Sentiment Breakdown",
    f"{_top_sentiment} is the most common review sentiment with {_top_sentiment_count:,} reviews. "
    "The breakdown provides a simple view of overall customer feedback."
)

(_REQUIRED_OUT / "Phase2_EDA_Observations.txt").write_text(
    "STREAMFLIX — PHASE 2 EDA OBSERVATIONS\n" + "=" * 70 + "\n\n" + "\n".join(_obs),
    encoding="utf-8"
)

# ---------------------------------------------------------------------------
# PHASE 3 — ALL 10 REQUIRED BUSINESS KPIs
# ---------------------------------------------------------------------------
print("\n" + "-" * 90)
print("PHASE 3 — ALL 10 REQUIRED BUSINESS KPIs")
print("-" * 90)

_total_subscribers = len(_req_sub)
_active_subscribers = int(_req_sub["is_active"].sum())
_inactive_subscribers = _total_subscribers - _active_subscribers
_total_watch_hours = float(_req_watch["watch_duration_min"].sum() / 60)
_avg_completion = float(_req_watch["completion_pct"].mean())
_mrr = float(_req_sub.loc[_req_sub["is_active"], "monthly_price_usd"].sum())
_arpu = float(_mrr / _active_subscribers) if _active_subscribers else 0.0
_avg_watch_hours_active_sub = float(_total_watch_hours / _active_subscribers) if _active_subscribers else 0.0
_watchlist_conversion = float(_req_watchlist["watched"].mean() * 100)

# Hit concentration: plays from top 10% titles / total plays
_plays_by_title = _req_watch["title_id"].value_counts().sort_values(ascending=False)
_top_n_titles = max(1, int(_math.ceil(len(_plays_by_title) * 0.10)))
_hit_concentration = float(_plays_by_title.head(_top_n_titles).sum() / len(_req_watch) * 100)

# Originals share of hours
_original_hours = float(
    _req_wh_titles.loc[_req_wh_titles["is_original"], "watch_duration_min"].sum() / 60
)
_originals_share = float(_original_hours / _total_watch_hours * 100) if _total_watch_hours else 0.0

_kpis = {
    "Total Watch Hours": _total_watch_hours,
    "Active Rate": _active_subscribers / _total_subscribers * 100 if _total_subscribers else 0.0,
    "Churn Rate": _inactive_subscribers / _total_subscribers * 100 if _total_subscribers else 0.0,
    "Avg Completion Rate": _avg_completion,
    "Monthly Recurring Revenue (MRR)": _mrr,
    "ARPU": _arpu,
    "Avg Watch Time / Active Subscriber (hours)": _avg_watch_hours_active_sub,
    "Watchlist Conversion": _watchlist_conversion,
    "Hit Concentration — Top 10% Titles": _hit_concentration,
    "Originals Share of Hours": _originals_share,
}

for _k, _v in _kpis.items():
    if _k in {"Monthly Recurring Revenue (MRR)", "ARPU"}:
        print(f"{_k}: ${_v:,.2f}")
    elif "Hours" in _k and "Originals" not in _k and "Concentration" not in _k:
        print(f"{_k}: {_v:,.2f}")
    else:
        print(f"{_k}: {_v:.2f}%" if _k not in {"Avg Watch Time / Active Subscriber (hours)"} else f"{_k}: {_v:,.2f}")

_kpi_df = pd.DataFrame([
    {"KPI": "Total Watch Hours", "Value": _total_watch_hours, "Unit": "hours"},
    {"KPI": "Active Rate", "Value": _active_subscribers / _total_subscribers * 100 if _total_subscribers else 0, "Unit": "%"},
    {"KPI": "Churn Rate", "Value": _inactive_subscribers / _total_subscribers * 100 if _total_subscribers else 0, "Unit": "%"},
    {"KPI": "Avg Completion Rate", "Value": _avg_completion, "Unit": "%"},
    {"KPI": "Monthly Recurring Revenue (MRR)", "Value": _mrr, "Unit": "USD/month"},
    {"KPI": "ARPU", "Value": _arpu, "Unit": "USD/active subscriber/month"},
    {"KPI": "Avg Watch Time / Active Subscriber", "Value": _avg_watch_hours_active_sub, "Unit": "hours"},
    {"KPI": "Watchlist Conversion", "Value": _watchlist_conversion, "Unit": "%"},
    {"KPI": "Hit Concentration — Top 10% Titles", "Value": _hit_concentration, "Unit": "% of plays"},
    {"KPI": "Originals Share of Hours", "Value": _originals_share, "Unit": "% of watch hours"},
])
_kpi_df.to_csv(_REQUIRED_OUT / "Phase3_KPIs.csv", index=False)

# ---------------------------------------------------------------------------
# PHASE 4 — ASSIGNMENT-ALIGNED STATIC 5-PAGE DASHBOARD OUTPUTS
# ---------------------------------------------------------------------------
# The PRD specifies 5 dashboard pages with filters/slicers. Python/matplotlib
# can produce the required page content, but interactive slicers are a Power BI
# / Excel feature. These pages are therefore static dashboard-style outputs.
# ---------------------------------------------------------------------------
print("\n" + "-" * 90)
print("PHASE 4 — 5 ASSIGNMENT-ALIGNED DASHBOARD PAGES")
print("-" * 90)

_DASH_OUT = _REQUIRED_OUT / "Phase4_Dashboard_Pages"
_DASH_OUT.mkdir(exist_ok=True)

# Page 1 — Engagement Overview
_fig = plt.figure(figsize=(16, 9))
_ax = _fig.add_subplot(2, 2, 1)
_ax.plot(_monthly_hours.index.astype(str), _monthly_hours.values)
_ax.set_title("Monthly Watch Hours")
_ax.tick_params(axis="x", rotation=45)
_ax.grid(True, alpha=.2)
_ax = _fig.add_subplot(2, 2, 2)
_ax.bar(["Subscribers", "Active", "Inactive"], [_total_subscribers, _active_subscribers, _inactive_subscribers])
_ax.set_title("Subscriber Health")
_ax.grid(True, axis="y", alpha=.2)
_ax = _fig.add_subplot(2, 2, 3)
_ax.bar(["Completion", "Churn"], [_avg_completion, _inactive_subscribers/_total_subscribers*100])
_ax.set_ylim(0, 100)
_ax.set_title("Completion vs Churn (%)")
_ax.grid(True, axis="y", alpha=.2)
_ax = _fig.add_subplot(2, 2, 4)
_ax.text(.02, .75, f"Total Watch Hours\n{_total_watch_hours:,.1f}", fontsize=18)
_ax.text(.02, .50, f"Active Rate\n{_active_subscribers/_total_subscribers*100:.1f}%", fontsize=18)
_ax.text(.02, .25, f"Churn Rate\n{_inactive_subscribers/_total_subscribers*100:.1f}%", fontsize=18)
_ax.axis("off")
_fig.suptitle("STREAMFLIX — PAGE 1: ENGAGEMENT OVERVIEW\nFilter/Slicer placeholder: All dates", fontsize=18)
_fig.tight_layout(rect=[0, 0, 1, .94])
_fig.savefig(_DASH_OUT / "Page1_Engagement_Overview.png", dpi=160, bbox_inches="tight")
plt.close(_fig)

# Page 2 — Content Performance
_fig = plt.figure(figsize=(16, 9))
_ax = _fig.add_subplot(2, 2, 1)
_top10_titles = _req_wh_titles.groupby("title_name")["watch_duration_min"].sum().div(60).sort_values(ascending=False).head(10).sort_values()
_ax.barh(_top10_titles.index.astype(str), _top10_titles.values)
_ax.set_title("Top 10 Titles by Watch Hours")
_ax = _fig.add_subplot(2, 2, 2)
_ax.barh(_completion_by_genre.sort_values().index.astype(str), _completion_by_genre.sort_values().values)
_ax.set_title("Completion Rate by Genre")
_ax.set_xlim(0, 100)
_ax = _fig.add_subplot(2, 2, 3)
_ax.barh(_genre_hours.head(10).sort_values().index.astype(str), _genre_hours.head(10).sort_values().values)
_ax.set_title("Watch Hours by Genre")
_ax = _fig.add_subplot(2, 2, 4)
_ax.text(.03, .65, f"Top Genre\n{_top_genre}", fontsize=20)
_ax.text(.03, .35, f"Top Genre Hours\n{_top_genre_hours:,.1f}", fontsize=20)
_ax.axis("off")
_fig.suptitle("STREAMFLIX — PAGE 2: CONTENT PERFORMANCE\nFilter/Slicer placeholder: Genre", fontsize=18)
_fig.tight_layout(rect=[0, 0, 1, .94])
_fig.savefig(_DASH_OUT / "Page2_Content_Performance.png", dpi=160, bbox_inches="tight")
plt.close(_fig)

# Page 3 — Subscriber Insights
_new_subs = _req_sub.groupby(_req_sub["signup_date"].dt.to_period("M")).size().sort_index()
_fig = plt.figure(figsize=(16, 9))
_ax = _fig.add_subplot(2, 2, 1)
_ax.bar(_plan_counts.index.astype(str), _plan_counts.values)
_ax.set_title("Plan Distribution")
_ax = _fig.add_subplot(2, 2, 2)
_ax.barh(_req_sub["country"].value_counts().head(10).sort_values().index.astype(str), _req_sub["country"].value_counts().head(10).sort_values().values)
_ax.set_title("Top 10 Subscriber Countries")
_ax = _fig.add_subplot(2, 2, 3)
_ax.plot(_new_subs.index.astype(str), _new_subs.values, marker="o")
_ax.set_title("New Subscribers per Month")
_ax.tick_params(axis="x", rotation=45)
_ax = _fig.add_subplot(2, 2, 4)
_ax.text(.03, .65, f"Largest Plan\n{_top_plan}", fontsize=20)
_ax.text(.03, .35, f"Largest Market\n{_req_sub['country'].value_counts().idxmax()}", fontsize=20)
_ax.axis("off")
_fig.suptitle("STREAMFLIX — PAGE 3: SUBSCRIBER INSIGHTS\nFilter/Slicer placeholder: Plan / Country", fontsize=18)
_fig.tight_layout(rect=[0, 0, 1, .94])
_fig.savefig(_DASH_OUT / "Page3_Subscriber_Insights.png", dpi=160, bbox_inches="tight")
plt.close(_fig)

# Page 4 — Experience
_fig = plt.figure(figsize=(16, 9))
_ax = _fig.add_subplot(2, 2, 1)
_ax.bar(_device_counts.index.astype(str), _device_counts.values)
_ax.set_title("Device Breakdown")
_ax.tick_params(axis="x", rotation=30)
_ax = _fig.add_subplot(2, 2, 2)
_ax.hist(_req_ratings["rating"].dropna(), bins=np.arange(.5, 5.6, 1), rwidth=.8)
_ax.set_title("Rating Distribution")
_ax.set_xticks([1, 2, 3, 4, 5])
_ax = _fig.add_subplot(2, 2, 3)
_ax.bar(_sentiment_counts.index, _sentiment_counts.values)
_ax.set_title("Review Sentiment")
_ax = _fig.add_subplot(2, 2, 4)
_ax.text(.03, .65, f"Top Device\n{_top_device}", fontsize=20)
_ax.text(.03, .35, f"Top Sentiment\n{_top_sentiment}", fontsize=20)
_ax.axis("off")
_fig.suptitle("STREAMFLIX — PAGE 4: EXPERIENCE\nFilter/Slicer placeholder: Device / Rating", fontsize=18)
_fig.tight_layout(rect=[0, 0, 1, .94])
_fig.savefig(_DASH_OUT / "Page4_Experience.png", dpi=160, bbox_inches="tight")
plt.close(_fig)

# Page 5 — Catalogue & Investment
_content_investment = _req_titles.groupby("primary_genre").agg(
    license_cost_usd=("license_cost_usd", "sum"),
    total_watch_hours=("total_watch_hours", "sum")
)
_content_investment["hours_per_1k"] = (
    _content_investment["total_watch_hours"] /
    (_content_investment["license_cost_usd"] / 1000)
)
_fig = plt.figure(figsize=(16, 9))
_ax = _fig.add_subplot(2, 2, 1)
_orig = int(_req_titles["is_original"].sum())
_licensed = len(_req_titles) - _orig
_ax.pie([_orig, _licensed], labels=["Original", "Licensed"], autopct="%1.1f%%", startangle=90)
_ax.set_title("Originals vs Licensed Titles")
_ax = _fig.add_subplot(2, 2, 2)
_eff = _content_investment["hours_per_1k"].sort_values(ascending=False).head(10).sort_values()
_ax.barh(_eff.index.astype(str), _eff.values)
_ax.set_title("Watch Hours per $1K Spend — Top Genres")
_ax = _fig.add_subplot(2, 2, 3)
_expiry = _req_titles.loc[_req_titles["license_expiry"].notna(), "license_expiry"].dt.to_period("M").value_counts().sort_index().head(12)
_ax.bar(_expiry.index.astype(str), _expiry.values)
_ax.set_title("Upcoming Licence Expiries")
_ax.tick_params(axis="x", rotation=45)
_ax = _fig.add_subplot(2, 2, 4)
_ax.text(.03, .65, f"Original titles\n{_orig:,}", fontsize=20)
_ax.text(.03, .35, f"Licensed titles\n{_licensed:,}", fontsize=20)
_ax.axis("off")
_fig.suptitle("STREAMFLIX — PAGE 5: CATALOGUE & INVESTMENT\nFilter/Slicer placeholder: Genre / Licence type", fontsize=18)
_fig.tight_layout(rect=[0, 0, 1, .94])
_fig.savefig(_DASH_OUT / "Page5_Catalogue_Investment.png", dpi=160, bbox_inches="tight")
plt.close(_fig)

# ---------------------------------------------------------------------------
# MANAGEMENT SUMMARY REPORT — DATA DRIVEN, PLAIN LANGUAGE
# ---------------------------------------------------------------------------
_report = []
_report.append("STREAMFLIX — MANAGEMENT SUMMARY REPORT")
_report.append("=" * 70)
_report.append("")
_report.append("EXECUTIVE SUMMARY")
_report.append("-" * 70)
_report.append(
    f"StreamFlix has {_total_subscribers:,} subscribers and generated approximately "
    f"{_total_watch_hours:,.1f} watch hours in the analysed viewing history. "
    f"The active subscriber rate is {_active_subscribers/_total_subscribers*100:.1f}% and the churn rate is "
    f"{_inactive_subscribers/_total_subscribers*100:.1f}%. Average completion is {_avg_completion:.1f}%, "
    f"while active subscribers generate an estimated ${_mrr:,.2f} in monthly recurring revenue. "
    f"The most watched genre is {_top_genre}, and the largest viewing device category is {_top_device}."
)
_report.append("")
_report.append("TOP 3 FINDINGS")
_report.append("-" * 70)
_report.append(f"1. {_top_genre} is the leading genre by watch time, with about {_top_genre_hours:,.1f} hours.")
_report.append(f"2. The platform's average completion rate is {_avg_completion:.1f}%, showing how much content viewers finish on average.")
_report.append(f"3. {_top_plan} is the largest subscriber plan, while {_top_device} is the most-used viewing device.")
_report.append("")
_report.append("RISKS IDENTIFIED")
_report.append("-" * 70)
_report.append(f"• Churn is {_inactive_subscribers/_total_subscribers*100:.1f}%; this should be watched against the project target of below 30%.")
_report.append(f"• The top 10% of titles account for {_hit_concentration:.1f}% of plays, indicating the degree of concentration in viewing demand.")
_report.append(f"• Originals contribute {_originals_share:.1f}% of watch hours, so investment decisions should compare this engagement with production/licensing cost.")
_report.append("")
_report.append("OPPORTUNITIES")
_report.append("-" * 70)
_report.append(f"• Invest in and promote high-performing genres such as {_top_genre} while testing adjacent genres with strong completion.")
_report.append(f"• Improve the viewing experience on {_top_device}, the most-used device category.")
_report.append("• Use watchlist conversion and completion behaviour to identify content that should receive stronger promotion.")
_report.append("")
_report.append("RECOMMENDATIONS")
_report.append("-" * 70)
_report.append("1. Focus retention campaigns on inactive or high-risk subscriber segments and measure changes in churn.")
_report.append("2. Prioritise content investment using both watch hours and watch-hours-per-$1K rather than popularity alone.")
_report.append("3. Promote titles with strong completion but lower visibility to spread engagement beyond the most concentrated hits.")
_report.append("4. Use the five dashboard pages for recurring management review, with date, genre, plan, device and licence filters implemented in Power BI or Excel.")

(_REQUIRED_OUT / "Phase4_Management_Summary_Report.txt").write_text(
    "\n".join(_report), encoding="utf-8"
)

print("\n" + "=" * 90)
print("REQUIRED ADDITIONS COMPLETED")
print(f"Outputs saved in: {_REQUIRED_OUT}")
print("1. Phase1_DataQuality_Report.txt")
print("2. Phase2_EDA_Charts/ — all 10 required charts")
print("3. Phase2_EDA_Observations.txt")
print("4. Phase3_KPIs.csv — all 10 required KPIs")
print("5. Phase4_Dashboard_Pages/ — 5 assignment-aligned static pages")
print("6. Phase4_Management_Summary_Report.txt")
print("=" * 90)


# =====================================================================
# STREAMFLIX — FINAL UPDATE PATCH
# Added after the original project code:
# 1) Column-wise missing-value percentage report
# 2) Assignment-aligned Data Quality Report
# 3) DOCX Management Report
#
# NOTE:
# Interactive slicers cannot be created with static Matplotlib figures.
# A separate Streamlit dashboard is supplied as:
# StreamFlix_Interactive_Dashboard.py
# =====================================================================

from pathlib import Path as _PatchPath
import pandas as _PatchPd
import numpy as _PatchNp

_PATCH_BASE = _PatchPath(__file__).resolve().parent
_PATCH_OUT = _PATCH_BASE / "StreamFlix_Final_Outputs"
_PATCH_OUT.mkdir(exist_ok=True)

def _patch_load_csv(filename):
    # Reuse the corrected global loader so this final validation section
    # cannot fall back to the old hard-coded/incorrect subscriber path.
    return load_project_csv(filename)

_patch_subscribers = _patch_load_csv("subscribers.csv")
_patch_titles = _patch_load_csv("titles.csv")
_patch_watch = _patch_load_csv("watch_history.csv")
_patch_ratings = _patch_load_csv("ratings.csv")
_patch_reviews = _patch_load_csv("reviews.csv")
_patch_watchlist = _patch_load_csv("watchlist.csv")

# -------------------------
# 1. Missing-value report
# -------------------------
_patch_tables = {
    "subscribers": _patch_subscribers,
    "titles": _patch_titles,
    "watch_history": _patch_watch,
    "ratings": _patch_ratings,
    "reviews": _patch_reviews,
    "watchlist": _patch_watchlist,
}

_patch_quality_rows = []
for _name, _table in _patch_tables.items():
    for _col in _table.columns:
        _null_count = int(_table[_col].isna().sum())
        _patch_quality_rows.append({
            "table": _name,
            "column": _col,
            "row_count": len(_table),
            "null_count": _null_count,
            "null_percentage": round((_null_count / len(_table)) * 100, 4),
            "dtype": str(_table[_col].dtype),
        })

_patch_missing_report = _PatchPd.DataFrame(_patch_quality_rows)
_patch_missing_report.to_csv(
    _PATCH_OUT / "Phase1_Missing_Value_Percentage_Report.csv",
    index=False
)

# -------------------------
# 2. Complete quality report
# -------------------------
for _c in ["signup_date", "churn_date"]:
    if _c in _patch_subscribers.columns:
        _patch_subscribers[_c] = _PatchPd.to_datetime(
            _patch_subscribers[_c], errors="coerce"
        )

for _c in ["watch_date"]:
    _patch_watch[_c] = _PatchPd.to_datetime(
        _patch_watch[_c], errors="coerce"
    )

for _c in ["rating_date"]:
    _patch_ratings[_c] = _PatchPd.to_datetime(
        _patch_ratings[_c], errors="coerce"
    )

for _c in ["review_date"]:
    _patch_reviews[_c] = _PatchPd.to_datetime(
        _patch_reviews[_c], errors="coerce"
    )

for _c in ["added_date"]:
    _patch_watchlist[_c] = _PatchPd.to_datetime(
        _patch_watchlist[_c], errors="coerce"
    )

_patch_subs_pk_dupes = int(_patch_subscribers["subscriber_id"].duplicated().sum())
_patch_titles_pk_dupes = int(_patch_titles["title_id"].duplicated().sum())
_patch_watch_pk_dupes = int(_patch_watch["watch_id"].duplicated().sum())
_patch_rating_pk_dupes = int(_patch_ratings["rating_id"].duplicated().sum())
_patch_review_pk_dupes = int(_patch_reviews["review_id"].duplicated().sum())
_patch_watchlist_pk_dupes = int(_patch_watchlist["watchlist_id"].duplicated().sum())

_patch_orphan_watch_sub = int(
    (~_patch_watch["subscriber_id"].isin(_patch_subscribers["subscriber_id"])).sum()
)
_patch_orphan_watch_title = int(
    (~_patch_watch["title_id"].isin(_patch_titles["title_id"])).sum()
)
_patch_orphan_wl_sub = int(
    (~_patch_watchlist["subscriber_id"].isin(_patch_subscribers["subscriber_id"])).sum()
)
_patch_orphan_wl_title = int(
    (~_patch_watchlist["title_id"].isin(_patch_titles["title_id"])).sum()
)

_patch_churn_bad = int(
    (
        _patch_subscribers["churn_date"].notna()
        & (
            _patch_subscribers["signup_date"].isna()
            | (_patch_subscribers["churn_date"] <= _patch_subscribers["signup_date"])
        )
    ).sum()
)

_patch_active_churn_mismatch = int(
    (
        _patch_subscribers["is_active"].eq(True)
        & _patch_subscribers["churn_date"].notna()
    ).sum()
)

_patch_watch_duration_outliers = int(
    (
        _patch_watch["watch_duration_min"]
        > _patch_watch["content_duration_min"]
    ).sum()
)

_patch_expected_completion = (
    _patch_watch["watch_duration_min"]
    / _patch_watch["content_duration_min"].replace(0, _PatchNp.nan)
    * 100
)
_patch_completion_gap = (
    _patch_watch["completion_pct"] - _patch_expected_completion
).abs()
_patch_completion_mismatch = int(
    (_patch_completion_gap > 2).fillna(False).sum()
)

_patch_valid_sentiments = {"Positive", "Neutral", "Negative"}
_patch_invalid_sentiment = int(
    (~_patch_reviews["sentiment"].isin(_patch_valid_sentiments)).sum()
)

_patch_report_lines = [
    "STREAMFLIX — DATA QUALITY REPORT",
    "=" * 72,
    "",
    "1. Missing values",
    "-" * 72,
]

for _name, _table in _patch_tables.items():
    _nulls = _table.isna().sum()
    _nonzero = _nulls[_nulls > 0]
    if _nonzero.empty:
        _patch_report_lines.append(
            f"{_name}: No missing values found."
        )
    else:
        _patch_report_lines.append(f"{_name}:")
        for _col, _count in _nonzero.items():
            _pct = (_count / len(_table)) * 100
            _patch_report_lines.append(
                f"  - {_col}: {_count:,} nulls ({_pct:.2f}%)"
            )

_patch_report_lines += [
    "",
    "2. Primary-key duplicate checks",
    "-" * 72,
    f"subscribers.subscriber_id: {_patch_subs_pk_dupes:,}",
    f"titles.title_id: {_patch_titles_pk_dupes:,}",
    f"watch_history.watch_id: {_patch_watch_pk_dupes:,}",
    f"ratings.rating_id: {_patch_rating_pk_dupes:,}",
    f"reviews.review_id: {_patch_review_pk_dupes:,}",
    f"watchlist.watchlist_id: {_patch_watchlist_pk_dupes:,}",
    "",
    "3. Referential integrity",
    "-" * 72,
    f"watch_history → subscribers orphans: {_patch_orphan_watch_sub:,}",
    f"watch_history → titles orphans: {_patch_orphan_watch_title:,}",
    f"watchlist → subscribers orphans: {_patch_orphan_wl_sub:,}",
    f"watchlist → titles orphans: {_patch_orphan_wl_title:,}",
    "",
    "4. Business-rule validation",
    "-" * 72,
    f"Churn date <= signup date: {_patch_churn_bad:,}",
    f"Active subscribers with churn_date populated: {_patch_active_churn_mismatch:,}",
    f"Watch sessions longer than content duration: {_patch_watch_duration_outliers:,}",
    f"Completion percentage mismatch (>2 percentage points): {_patch_completion_mismatch:,}",
    f"Invalid review sentiment values: {_patch_invalid_sentiment:,}",
    "",
    "5. Overall conclusion",
    "-" * 72,
    "The report now includes column-wise missing counts and percentages,",
    "primary-key checks, referential-integrity checks, and the assignment's",
    "required business-rule validations.",
]

(_PATCH_OUT / "Phase1_DataQuality_Report.txt").write_text(
    "\n".join(_patch_report_lines),
    encoding="utf-8"
)

print("\nUPDATED PHASE 1 OUTPUTS")
print(_PATCH_OUT / "Phase1_Missing_Value_Percentage_Report.csv")
print(_PATCH_OUT / "Phase1_DataQuality_Report.txt")
