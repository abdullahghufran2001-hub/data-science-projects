import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Universal Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# LOGIN SETTINGS
# ==================================================

USERNAME = "admin"
PASSWORD = "admin123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.header{
    background:linear-gradient(90deg,#0f172a,#2563eb);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOGIN PAGE
# ==================================================

if not st.session_state.logged_in:

    st.markdown("""
    <div class="header">
        <h1>🔐 Universal Analytics Dashboard</h1>
        <p>Login to Continue</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login", use_container_width=True):

        if username == USERNAME and password == PASSWORD:

            st.session_state.logged_in = True
            st.success("✅ Login Successful")
            st.rerun()

        else:

            st.error(
                "❌ Sorry! We cannot open this site. Invalid Username or Password."
            )

    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.success("✅ Logged In")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="header">
<h1>📊 Universal Data Analytics Dashboard</h1>
<p>Upload Any CSV or Excel File and Get Instant Insights</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# FILE UPLOADER
# ==================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# ==================================================
# ANALYSIS
# ==================================================

if uploaded_file:

    try:

        # Load Data

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File Uploaded Successfully")

        # ==================================================
        # DATASET OVERVIEW
        # ==================================================

        rows = df.shape[0]
        cols = df.shape[1]
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📄 Rows", f"{rows:,}")
        c2.metric("📊 Columns", f"{cols:,}")
        c3.metric("⚠ Missing", f"{missing:,}")
        c4.metric("🔁 Duplicates", f"{duplicates:,}")

        st.divider()

        # ==================================================
        # COLUMN INFORMATION
        # ==================================================

        st.subheader("📋 Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # DATA PREVIEW
        # ==================================================

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # STATISTICAL SUMMARY
        # ==================================================

        st.subheader("📈 Statistical Summary")

        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # COLUMN DETECTION
        # ==================================================

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns.tolist()

        cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # ==================================================
        # NUMERIC ANALYSIS
        # ==================================================

        if len(numeric_cols) > 0:

            st.subheader("📊 Numeric Analysis")

            selected_num = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            col1, col2 = st.columns(2)

            with col1:

                fig_hist = px.histogram(
                    df,
                    x=selected_num,
                    nbins=25,
                    title=f"Distribution of {selected_num}"
                )

                st.plotly_chart(
                    fig_hist,
                    use_container_width=True
                )

            with col2:

                fig_box = px.box(
                    df,
                    y=selected_num,
                    title=f"Box Plot of {selected_num}"
                )

                st.plotly_chart(
                    fig_box,
                    use_container_width=True
                )

        # ==================================================
        # CATEGORY ANALYSIS
        # ==================================================

        if len(cat_cols) > 0:

            st.subheader("📌 Category Analysis")

            selected_cat = st.selectbox(
                "Select Category Column",
                cat_cols
            )

            top_cat = (
                df[selected_cat]
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )

            top_cat.columns = [
                selected_cat,
                "Count"
            ]

            fig_bar = px.bar(
                top_cat,
                x=selected_cat,
                y="Count",
                title=f"Top Categories in {selected_cat}"
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

            fig_pie = px.pie(
                top_cat,
                names=selected_cat,
                values="Count",
                title=f"{selected_cat} Distribution"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        # ==================================================
        # CORRELATION HEATMAP
        # ==================================================

        if len(numeric_cols) > 1:

            st.subheader("🔥 Correlation Heatmap")

            corr = df[numeric_cols].corr()

            heatmap = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.columns),
                annotation_text=round(corr, 2).values,
                showscale=True
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True
            )

        # ==================================================
        # MISSING VALUES
        # ==================================================

        st.subheader("⚠ Missing Values Report")

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        # ==================================================
        # DOWNLOAD REPORT
        # ==================================================

        summary = pd.DataFrame({
            "Metric": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows"
            ],
            "Value": [
                rows,
                cols,
                missing,
                duplicates
            ]
        })

        csv = summary.to_csv(index=False)

        st.download_button(
            label="⬇ Download Summary Report",
            data=csv,
            file_name="dataset_summary.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"❌ Error Processing File: {e}")

else:

    st.info(
        "📂 Upload Any CSV or Excel File To Start Analysis"
    )
