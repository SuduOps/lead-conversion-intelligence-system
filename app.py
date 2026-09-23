# =============================================================================
# Lead Conversion Intelligence System
# Capstone Project – X Education Lead Scoring Dataset
# =============================================================================
# This Streamlit app performs four-tier analytics (Descriptive, Diagnostic,
# Predictive, Prescriptive) on historical marketing leads to help sales teams
# prioritise follow-ups and maximise conversion rates.
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.impute import SimpleImputer

import io

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lead Conversion Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for KPI cards ─────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background-color: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1f2328; }
    .kpi-label { font-size: 0.85rem; color: #57606a; margin-top: 4px; }
    .insight-box {
        background-color: #eef4ff;
        border-left: 4px solid #3b82d4;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.93rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING  (cached so it runs only once per session)
# =============================================================================

DATA_PATH = "data/leads.csv"

@st.cache_data(show_spinner=False)
def load_raw_data():
    """Load the raw CSV without any modifications."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df, None
    except FileNotFoundError:
        return None, f"❌ Dataset not found at '{DATA_PATH}'. Make sure 'data/leads.csv' exists."
    except Exception as e:
        return None, f"❌ Could not load dataset: {e}"


@st.cache_data(show_spinner=False)
def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a COPY of the raw dataframe.
    The original file on disk is never touched.

    Steps
    -----
    1. Drop exact duplicate rows.
    2. Replace the sentinel value "Select" with NaN throughout.
    3. Fill missing numeric values with the column median.
    4. Fill missing categorical values with "Unknown" (or mode for high-cardinality columns).
    5. Strip leading/trailing whitespace from string columns.
    6. Normalise casing of object columns to Title Case where sensible.
    """
    df = raw_df.copy()

    # Step 1 – Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Step 2 – Replace "Select" (a web-form default) with NaN
    df.replace("Select", np.nan, inplace=True)

    # Step 3 – Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target / identifier columns from imputation
    numeric_to_impute = [c for c in numeric_cols
                         if c not in ("Converted", "Lead Number")]
    for col in numeric_to_impute:
        df[col].fillna(df[col].median(), inplace=True)

    # Step 4 – Fill missing categorical values
    # High-cardinality free-text columns → "Unknown"
    high_null_thresh = 0.20          # columns with >20% missing → "Unknown"
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    exclude_id_cols = ["Prospect ID"]
    for col in cat_cols:
        if col in exclude_id_cols:
            continue
        null_frac = df[col].isnull().mean()
        if null_frac > high_null_thresh:
            df[col].fillna("Unknown", inplace=True)
        else:
            # Fill with mode (most frequent value)
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

    # Step 5 – Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Step 6 – Normalise Lead Source casing (e.g. "google" → "Google")
    if "Lead Source" in df.columns:
        df["Lead Source"] = df["Lead Source"].str.title()

    return df


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def kpi_card(value, label):
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# =============================================================================
# MODEL TRAINING  (cached so the model is built only once)
# =============================================================================

@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """
    Train Logistic Regression and Random Forest on the cleaned dataframe.
    Returns a dict with both models, metrics, and the test split.
    """
    TARGET = "Converted"
    DROP_COLS = ["Prospect ID", "Lead Number", TARGET]

    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET].astype(int)

    # Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = {}
    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
    ]:
        pipe = Pipeline([("pre", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "pipeline": pipe,
            "y_test":   y_test,
            "y_pred":   y_pred,
            "y_proba":  y_proba,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "cm":        confusion_matrix(y_test, y_pred),
        }

    # Determine best model by ROC-AUC
    best = max(results, key=lambda k: results[k]["roc_auc"])
    results["best_model_name"] = best
    results["best_pipeline"]   = results[best]["pipeline"]
    results["X_columns"]       = X.columns.tolist()
    results["num_cols"]        = num_cols
    results["cat_cols"]        = cat_cols

    return results


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset …"):
        raw_df, err = load_raw_data()

    if err:
        st.error(err)
        st.stop()

    with st.spinner("Cleaning data …"):
        df = clean_data(raw_df)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.image(
        "https://img.icons8.com/color/96/target.png",
        width=60,
    )
    st.sidebar.title("🎯 Lead Intelligence")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigate to",
        ["📋 Data Overview",
         "📊 Descriptive Analytics",
         "🔍 Diagnostic Analytics",
         "🤖 Predictive Analytics",
         "🎯 Prescriptive Analytics"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filters")

    # Build sidebar filters dynamically for key categorical columns
    filter_cols = {
        "Lead Source":                   "Lead Source",
        "Lead Origin":                   "Lead Origin",
        "Last Activity":                 "Last Activity",
        "What is your current occupation": "Occupation",
        "City":                          "City",
    }

    filtered_df = df.copy()
    for col, label in filter_cols.items():
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            selected = st.sidebar.multiselect(label, options, default=[])
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: X Education Lead Scoring Dataset")

    # ── App title ─────────────────────────────────────────────────────────────
    st.title("🎯 Lead Conversion Intelligence System")
    st.caption("Analyse · Diagnose · Predict · Prescribe — powered by X Education lead data")
    st.markdown("---")

    # ── KPI Row (always visible) ──────────────────────────────────────────────
    total   = len(filtered_df)
    conv    = int(filtered_df["Converted"].sum())
    non_conv = total - conv
    rate    = conv / total * 100 if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(f"{total:,}", "Total Leads"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(f"{conv:,}", "Converted Leads"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(f"{non_conv:,}", "Non-Converted Leads"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(f"{rate:.1f}%", "Conversion Rate"), unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================================
    # PAGE 1 – Data Overview
    # =========================================================================
    if page == "📋 Data Overview":
        st.header("📋 Data Quality & Overview")

        tab1, tab2, tab3 = st.tabs(["Raw Dataset Info", "Cleaned Dataset Info", "Cleaning Steps"])

        with tab1:
            st.subheader("Raw Dataset")
            r, c = raw_df.shape
            st.markdown(f"**Rows:** {r:,}  |  **Columns:** {c}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Column Names & Data Types**")
                dtype_df = pd.DataFrame({
                    "Column": raw_df.dtypes.index,
                    "Type":   raw_df.dtypes.astype(str).values,
                })
                st.dataframe(dtype_df, use_container_width=True, height=400)
            with col_b:
                st.markdown("**Missing Values Per Column**")
                miss = raw_df.isnull().sum().reset_index()
                miss.columns = ["Column", "Missing Count"]
                miss["Missing %"] = (miss["Missing Count"] / r * 100).round(2)
                miss = miss[miss["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
                st.dataframe(miss, use_container_width=True, height=400)

            st.markdown(f"**Duplicate Rows:** {raw_df.duplicated().sum():,}")
            st.subheader("Descriptive Statistics (numeric columns)")
            st.dataframe(raw_df.describe().T.round(2), use_container_width=True)

        with tab2:
            st.subheader("Cleaned Dataset")
            r2, c2 = df.shape
            st.markdown(f"**Rows:** {r2:,}  |  **Columns:** {c2}")
            st.markdown(f"**Remaining Duplicate Rows:** {df.duplicated().sum():,}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Missing Values After Cleaning**")
                miss2 = df.isnull().sum().reset_index()
                miss2.columns = ["Column", "Missing Count"]
                miss2 = miss2[miss2["Missing Count"] > 0]
                if miss2.empty:
                    st.success("✅ No missing values remain after cleaning.")
                else:
                    st.dataframe(miss2, use_container_width=True)
            with col_b:
                st.markdown("**Cleaned Data Sample (first 5 rows)**")
                st.dataframe(df.head(), use_container_width=True)

            st.subheader("Descriptive Statistics (cleaned numeric columns)")
            st.dataframe(df.describe().T.round(2), use_container_width=True)

        with tab3:
            st.subheader("Data Cleaning Steps Explained")
            steps = [
                ("1. Removed Duplicate Rows",
                 "Exact duplicate records were identified and dropped to prevent any lead being counted more than once during analysis or model training."),
                ("2. Replaced 'Select' with Missing (NaN)",
                 "'Select' is the default placeholder value left by users who did not choose an option on the web form. Treating it as a real category would distort analysis, so it is replaced with NaN before imputation."),
                ("3. Numeric Imputation – Median",
                 "Columns such as TotalVisits and Page Views Per Visit have a small fraction of missing values. The median is used instead of the mean because it is robust to the outliers common in web-traffic data."),
                ("4. Categorical Imputation – 'Unknown' or Mode",
                 "Columns with more than 20% missing values (e.g. Tags, Lead Quality, What matters most …) are filled with the label 'Unknown', which preserves the information that the value was absent. Columns with fewer missing values are filled with the mode (the most frequent observed category), which minimises distributional distortion."),
                ("5. Whitespace Standardisation",
                 "Leading and trailing spaces were stripped from all string columns to prevent identical categories from being treated as different values (e.g. 'Mumbai ' vs 'Mumbai')."),
                ("6. Lead Source Casing Normalisation",
                 "Values such as 'google' and 'Google' represent the same channel. All Lead Source values were converted to Title Case to merge these duplicates."),
                ("7. Original File Preserved",
                 "All cleaning is applied to an in-memory copy of the data. The file data/leads.csv is never modified, moved, or deleted."),
            ]
            for title, desc in steps:
                with st.expander(title, expanded=False):
                    st.write(desc)

    # =========================================================================
    # PAGE 2 – Descriptive Analytics
    # =========================================================================
    elif page == "📊 Descriptive Analytics":
        st.header("📊 Descriptive Analytics — What Happened?")
        st.markdown("Understand the shape of your lead data and overall conversion patterns.")

        tab1, tab2, tab3 = st.tabs(["Conversion Overview", "Channel & Source Breakdown", "Behaviour Distributions"])

        with tab1:
            st.subheader("Overall Conversion Split")
            pie_data = filtered_df["Converted"].map({1: "Converted", 0: "Not Converted"}).value_counts().reset_index()
            pie_data.columns = ["Status", "Count"]
            fig_pie = px.pie(
                pie_data, values="Count", names="Status",
                color="Status",
                color_discrete_map={"Converted": "#3b82d4", "Not Converted": "#e5e7eb"},
                title="Overall Lead Conversion Split",
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Conversion rate over Lead Origin
            st.subheader("Conversion Rate by Lead Origin")
            lo_rate = (filtered_df.groupby("Lead Origin")["Converted"]
                       .agg(["sum", "count"])
                       .rename(columns={"sum": "Converted", "count": "Total"})
                       .reset_index())
            lo_rate["Conversion Rate (%)"] = (lo_rate["Converted"] / lo_rate["Total"] * 100).round(1)
            lo_rate = lo_rate.sort_values("Conversion Rate (%)", ascending=False)
            fig_lo = px.bar(
                lo_rate, x="Lead Origin", y="Conversion Rate (%)",
                color="Conversion Rate (%)", color_continuous_scale="Blues",
                text="Conversion Rate (%)",
                title="Conversion Rate by Lead Origin",
            )
            fig_lo.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_lo.update_layout(showlegend=False)
            st.plotly_chart(fig_lo, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)

            # Lead Source
            with col_a:
                st.subheader("Conversion Rate by Lead Source")
                ls_rate = (filtered_df.groupby("Lead Source")["Converted"]
                           .agg(["sum", "count"])
                           .rename(columns={"sum": "Converted", "count": "Total"})
                           .reset_index())
                ls_rate["Conversion Rate (%)"] = (ls_rate["Converted"] / ls_rate["Total"] * 100).round(1)
                ls_rate = ls_rate[ls_rate["Total"] >= 10].sort_values("Conversion Rate (%)", ascending=True)
                fig_ls = px.bar(
                    ls_rate, y="Lead Source", x="Conversion Rate (%)",
                    orientation="h", color="Conversion Rate (%)",
                    color_continuous_scale="teal",
                    title="Conversion Rate by Lead Source (min. 10 leads)",
                )
                st.plotly_chart(fig_ls, use_container_width=True)

            # Last Activity
            with col_b:
                st.subheader("Conversion Rate by Last Activity")
                la_rate = (filtered_df.groupby("Last Activity")["Converted"]
                           .agg(["sum", "count"])
                           .rename(columns={"sum": "Converted", "count": "Total"})
                           .reset_index())
                la_rate["Conversion Rate (%)"] = (la_rate["Converted"] / la_rate["Total"] * 100).round(1)
                la_rate = la_rate[la_rate["Total"] >= 5].sort_values("Conversion Rate (%)", ascending=True)
                fig_la = px.bar(
                    la_rate, y="Last Activity", x="Conversion Rate (%)",
                    orientation="h", color="Conversion Rate (%)",
                    color_continuous_scale="Oranges",
                    title="Conversion Rate by Last Activity (min. 5 leads)",
                )
                st.plotly_chart(fig_la, use_container_width=True)

            # Occupation
            st.subheader("Conversion Rate by Occupation")
            occ_col = "What is your current occupation"
            if occ_col in filtered_df.columns:
                occ_rate = (filtered_df.groupby(occ_col)["Converted"]
                            .agg(["sum", "count"])
                            .rename(columns={"sum": "Converted", "count": "Total"})
                            .reset_index())
                occ_rate["Conversion Rate (%)"] = (occ_rate["Converted"] / occ_rate["Total"] * 100).round(1)
                occ_rate = occ_rate.sort_values("Conversion Rate (%)", ascending=False)
                fig_occ = px.bar(
                    occ_rate, x=occ_col, y="Conversion Rate (%)",
                    color="Conversion Rate (%)", color_continuous_scale="Purples",
                    text="Conversion Rate (%)",
                    title="Conversion Rate by Occupation",
                    labels={occ_col: "Occupation"},
                )
                fig_occ.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_occ, use_container_width=True)

            # City
            if "City" in filtered_df.columns:
                st.subheader("Lead Volume by City")
                city_counts = filtered_df["City"].value_counts().reset_index()
                city_counts.columns = ["City", "Count"]
                fig_city = px.bar(
                    city_counts.head(10), x="City", y="Count",
                    color="Count", color_continuous_scale="Blues",
                    title="Top Cities by Lead Volume",
                )
                st.plotly_chart(fig_city, use_container_width=True)

        with tab3:
            st.subheader("Distribution of Total Visits")
            fig_tv = px.histogram(
                filtered_df, x="TotalVisits", nbins=40,
                color_discrete_sequence=["#3b82d4"],
                title="Distribution of Total Website Visits per Lead",
                labels={"TotalVisits": "Total Visits"},
            )
            fig_tv.update_layout(bargap=0.05)
            st.plotly_chart(fig_tv, use_container_width=True)

            st.subheader("Distribution of Total Time Spent on Website")
            fig_ts = px.histogram(
                filtered_df, x="Total Time Spent on Website", nbins=40,
                color_discrete_sequence=["#7c5cd8"],
                title="Distribution of Total Time Spent on Website (minutes)",
                labels={"Total Time Spent on Website": "Time Spent (mins)"},
            )
            fig_ts.update_layout(bargap=0.05)
            st.plotly_chart(fig_ts, use_container_width=True)

            st.subheader("Page Views Per Visit Distribution")
            fig_pv = px.histogram(
                filtered_df, x="Page Views Per Visit", nbins=30,
                color_discrete_sequence=["#f59e0b"],
                title="Distribution of Page Views Per Visit",
            )
            st.plotly_chart(fig_pv, use_container_width=True)

    # =========================================================================
    # PAGE 3 – Diagnostic Analytics
    # =========================================================================
    elif page == "🔍 Diagnostic Analytics":
        st.header("🔍 Diagnostic Analytics — Why Did It Happen?")
        st.markdown("Compare converted vs non-converted leads to uncover the key drivers of conversion.")

        tab1, tab2, tab3 = st.tabs(["Numeric Comparisons", "Categorical Comparisons", "Business Insights"])

        with tab1:
            st.subheader("Numeric Feature Distributions by Conversion Status")
            numeric_feats = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
                             "Asymmetrique Activity Score", "Asymmetrique Profile Score"]
            numeric_feats = [f for f in numeric_feats if f in filtered_df.columns]

            label_map = {0: "Not Converted", 1: "Converted"}
            plot_df = filtered_df.copy()
            plot_df["Status"] = plot_df["Converted"].map(label_map)

            for feat in numeric_feats:
                fig = px.box(
                    plot_df, x="Status", y=feat, color="Status",
                    color_discrete_map={"Converted": "#3b82d4", "Not Converted": "#e5e7eb"},
                    title=f"{feat} — Converted vs Not Converted",
                    points="outliers",
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Summary stats table
            st.subheader("Statistical Summary by Conversion Status")
            summary = plot_df.groupby("Status")[numeric_feats].median().T.round(2)
            summary.columns.name = None
            st.dataframe(summary, use_container_width=True)

        with tab2:
            st.subheader("Categorical Feature Breakdown by Conversion Status")
            cat_feats = ["Lead Source", "Lead Origin", "Last Activity",
                         "What is your current occupation", "City",
                         "What matters most to you in choosing a course"]
            cat_feats = [f for f in cat_feats if f in filtered_df.columns]

            for feat in cat_feats:
                grp = (filtered_df.groupby([feat, "Converted"])
                       .size().reset_index(name="Count"))
                grp["Conversion"] = grp["Converted"].map({1: "Converted", 0: "Not Converted"})
                # Keep only top 10 categories by total volume
                top_cats = grp.groupby(feat)["Count"].sum().nlargest(10).index
                grp = grp[grp[feat].isin(top_cats)]
                fig = px.bar(
                    grp, x=feat, y="Count", color="Conversion",
                    barmode="group",
                    color_discrete_map={"Converted": "#3b82d4", "Not Converted": "#e5e7eb"},
                    title=f"Lead Count by {feat} and Conversion Status",
                )
                fig.update_layout(xaxis_tickangle=-30, legend_title_text="Status")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Key Business Insights")

            # Insight 1 – Time on website
            conv_time   = filtered_df[filtered_df["Converted"] == 1]["Total Time Spent on Website"].median()
            noconv_time = filtered_df[filtered_df["Converted"] == 0]["Total Time Spent on Website"].median()
            insight(
                f"**Time on Website:** Converted leads spend a median of **{conv_time:.0f} minutes** on the "
                f"website, compared to only **{noconv_time:.0f} minutes** for non-converted leads. "
                f"Leads that spend more time exploring content are significantly more likely to convert."
            )

            # Insight 2 – Total Visits
            conv_visits   = filtered_df[filtered_df["Converted"] == 1]["TotalVisits"].median()
            noconv_visits = filtered_df[filtered_df["Converted"] == 0]["TotalVisits"].median()
            insight(
                f"**Website Visits:** Converted leads make a median of **{conv_visits:.0f} visits**, "
                f"versus **{noconv_visits:.0f} visits** for non-converted leads. "
                f"Re-targeting leads who have visited multiple times is a high-priority strategy."
            )

            # Insight 3 – Lead Source
            if "Lead Source" in filtered_df.columns:
                ls_conv = (filtered_df.groupby("Lead Source")["Converted"]
                           .agg(["sum", "count"])
                           .eval("rate = sum / count")
                           .sort_values("rate", ascending=False))
                best_src = ls_conv.index[0]
                best_rate = ls_conv["rate"].iloc[0] * 100
                insight(
                    f"**Best Lead Source:** '{best_src}' has the highest conversion rate at "
                    f"**{best_rate:.1f}%**. Marketing budgets should prioritise this channel."
                )

            # Insight 4 – Occupation
            occ_col = "What is your current occupation"
            if occ_col in filtered_df.columns:
                occ_conv = (filtered_df.groupby(occ_col)["Converted"]
                            .agg(["sum", "count"])
                            .eval("rate = sum / count")
                            .sort_values("rate", ascending=False))
                best_occ = occ_conv.index[0]
                best_occ_rate = occ_conv["rate"].iloc[0] * 100
                insight(
                    f"**Best Occupation Segment:** '{best_occ}' leads convert at "
                    f"**{best_occ_rate:.1f}%**. Sales messaging tailored to this audience can improve ROI."
                )

            # Insight 5 – Last Activity
            if "Last Activity" in filtered_df.columns:
                la_conv = (filtered_df.groupby("Last Activity")["Converted"]
                           .agg(["sum", "count"])
                           .query("count >= 5")
                           .eval("rate = sum / count")
                           .sort_values("rate", ascending=False))
                if not la_conv.empty:
                    best_act = la_conv.index[0]
                    best_act_rate = la_conv["rate"].iloc[0] * 100
                    insight(
                        f"**Engagement Trigger:** Leads whose last activity was '{best_act}' "
                        f"convert at **{best_act_rate:.1f}%**. "
                        f"Sales teams should prioritise follow-up within 24 hours of this activity."
                    )

    # =========================================================================
    # PAGE 4 – Predictive Analytics
    # =========================================================================
    elif page == "🤖 Predictive Analytics":
        st.header("🤖 Predictive Analytics — What Is Likely to Happen?")
        st.markdown(
            "Two machine-learning models are trained to predict which leads will convert. "
            "The better model is selected based on ROC-AUC score."
        )

        with st.spinner("Training models (this may take ~30 seconds on first load) …"):
            results = train_models(df)  # use full cleaned dataset for training

        best_name = results["best_model_name"]
        st.success(f"✅ Best model selected: **{best_name}** (highest ROC-AUC)")

        # ── Model Comparison Table ────────────────────────────────────────────
        st.subheader("Model Comparison")
        metrics_rows = []
        for name in ["Logistic Regression", "Random Forest"]:
            r = results[name]
            metrics_rows.append({
                "Model":     name,
                "Accuracy":  f"{r['accuracy']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall":    f"{r['recall']:.4f}",
                "F1-Score":  f"{r['f1']:.4f}",
                "ROC-AUC":   f"{r['roc_auc']:.4f}",
                "Selected":  "✅" if name == best_name else "",
            })
        cmp_df = pd.DataFrame(metrics_rows)
        st.dataframe(cmp_df.set_index("Model"), use_container_width=True)

        # ── Per-model detail ──────────────────────────────────────────────────
        for model_name in ["Logistic Regression", "Random Forest"]:
            r = results[model_name]
            with st.expander(f"{'⭐ ' if model_name == best_name else ''}Details — {model_name}", expanded=(model_name == best_name)):

                col_a, col_b = st.columns(2)

                # Confusion Matrix
                with col_a:
                    cm = r["cm"]
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        x=["Predicted: No", "Predicted: Yes"],
                        y=["Actual: No",    "Actual: Yes"],
                        title=f"Confusion Matrix — {model_name}",
                    )
                    fig_cm.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ROC Curve
                with col_b:
                    fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"ROC (AUC={r['roc_auc']:.3f})",
                        line=dict(color="#3b82d4", width=2),
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Random Classifier",
                        line=dict(color="#e5e7eb", dash="dash"),
                    ))
                    fig_roc.update_layout(
                        title=f"ROC Curve — {model_name}",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        legend=dict(x=0.6, y=0.1),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                # Feature importance / coefficients
                st.subheader("Top Feature Importances / Coefficients")
                pipeline     = r["pipeline"]
                preprocessor = pipeline.named_steps["pre"]
                classifier   = pipeline.named_steps["model"]

                # Reconstruct feature names after OHE
                num_feature_names = results["num_cols"]
                try:
                    ohe        = preprocessor.named_transformers_["cat"].named_steps["encoder"]
                    cat_names  = ohe.get_feature_names_out(results["cat_cols"]).tolist()
                except Exception:
                    cat_names = []
                all_feature_names = num_feature_names + cat_names

                if model_name == "Random Forest":
                    importances = classifier.feature_importances_
                    feat_df = pd.DataFrame({
                        "Feature":    all_feature_names,
                        "Importance": importances,
                    }).sort_values("Importance", ascending=False).head(20)
                    fig_fi = px.bar(
                        feat_df, y="Feature", x="Importance",
                        orientation="h", color="Importance",
                        color_continuous_scale="Blues",
                        title="Top 20 Feature Importances (Random Forest)",
                    )
                    fig_fi.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_fi, use_container_width=True)

                else:  # Logistic Regression coefficients
                    coefs = classifier.coef_[0]
                    coef_df = pd.DataFrame({
                        "Feature":     all_feature_names,
                        "Coefficient": coefs,
                    })
                    coef_df["Abs"] = coef_df["Coefficient"].abs()
                    coef_df = coef_df.sort_values("Abs", ascending=False).head(20)
                    coef_df["Direction"] = coef_df["Coefficient"].apply(
                        lambda x: "Positive (↑ Conversion)" if x > 0 else "Negative (↓ Conversion)"
                    )
                    fig_coef = px.bar(
                        coef_df.sort_values("Coefficient"), y="Feature", x="Coefficient",
                        orientation="h", color="Direction",
                        color_discrete_map={
                            "Positive (↑ Conversion)": "#3b82d4",
                            "Negative (↓ Conversion)": "#ef4444",
                        },
                        title="Top 20 Logistic Regression Coefficients",
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)

    # =========================================================================
    # PAGE 5 – Prescriptive Analytics
    # =========================================================================
    elif page == "🎯 Prescriptive Analytics":
        st.header("🎯 Prescriptive Analytics — What Should We Do?")
        st.markdown(
            "Every lead in the dataset is scored by the best model. "
            "Leads are then categorised and actionable recommendations are provided."
        )

        with st.spinner("Scoring all leads …"):
            results = train_models(df)

        best_pipeline = results["best_pipeline"]
        best_name     = results["best_model_name"]

        # Score all leads
        DROP_COLS = ["Prospect ID", "Lead Number", "Converted"]
        X_all = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        proba_all = best_pipeline.predict_proba(X_all)[:, 1]

        priority_df = df[["Prospect ID", "Lead Number", "Lead Source",
                           "Lead Origin", "City",
                           "What is your current occupation",
                           "Last Activity", "Converted"]].copy()
        priority_df.rename(columns={"What is your current occupation": "Occupation"}, inplace=True)
        priority_df["Conversion Probability (%)"] = (proba_all * 100).round(1)
        priority_df["Lead Category"] = pd.cut(
            proba_all,
            bins=[-0.01, 0.40, 0.70, 1.01],
            labels=["🥶 Cold Lead", "🌡️ Warm Lead", "🔥 Hot Lead"],
        )

        # ── Category summary ─────────────────────────────────────────────────
        st.subheader("Lead Priority Distribution")
        cat_counts = priority_df["Lead Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Colour"] = cat_counts["Category"].map({
            "🔥 Hot Lead":  "#ef4444",
            "🌡️ Warm Lead": "#f59e0b",
            "🥶 Cold Lead": "#3b82d4",
        })
        fig_cat = px.pie(
            cat_counts, values="Count", names="Category",
            color="Category",
            color_discrete_map={
                "🔥 Hot Lead":  "#ef4444",
                "🌡️ Warm Lead": "#f59e0b",
                "🥶 Cold Lead": "#3b82d4",
            },
            title=f"Lead Priority Split — Scored by {best_name}",
        )
        fig_cat.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_cat, use_container_width=True)

        # ── Probability histogram ─────────────────────────────────────────────
        st.subheader("Distribution of Predicted Conversion Probabilities")
        fig_prob = px.histogram(
            priority_df, x="Conversion Probability (%)",
            nbins=40, color_discrete_sequence=["#3b82d4"],
            title="Distribution of Predicted Conversion Probability (%)",
        )
        fig_prob.add_vline(x=40, line_dash="dash", line_color="#f59e0b",
                           annotation_text="Warm threshold (40%)")
        fig_prob.add_vline(x=70, line_dash="dash", line_color="#ef4444",
                           annotation_text="Hot threshold (70%)")
        st.plotly_chart(fig_prob, use_container_width=True)

        # ── Lead Priority Table ───────────────────────────────────────────────
        st.subheader("Lead Priority Table")
        st.markdown("Filter by lead category to find the right leads to act on first.")

        cat_filter = st.selectbox(
            "Filter by Category",
            ["All", "🔥 Hot Lead", "🌡️ Warm Lead", "🥶 Cold Lead"],
        )
        display_df = priority_df if cat_filter == "All" else priority_df[priority_df["Lead Category"] == cat_filter]
        display_df = display_df.sort_values("Conversion Probability (%)", ascending=False)

        st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=400)

        # ── Recommendations ───────────────────────────────────────────────────
        st.subheader("📋 Actionable Recommendations")

        hot_count  = (priority_df["Lead Category"] == "🔥 Hot Lead").sum()
        warm_count = (priority_df["Lead Category"] == "🌡️ Warm Lead").sum()
        cold_count = (priority_df["Lead Category"] == "🥶 Cold Lead").sum()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style='background:#fff1f0;border:1px solid #ef4444;border-radius:8px;padding:16px;'>
            <h4 style='color:#ef4444;'>🔥 Hot Leads ({hot_count:,})</h4>
            <b>Priority: Immediate Action</b><br><br>
            • Assign to senior sales reps within <b>24 hours</b>.<br>
            • Offer a <b>1-on-1 demo or consultation</b>.<br>
            • Use personalised outreach referencing their browsing behaviour.<br>
            • Offer limited-time enrollment discounts to close quickly.<br>
            • Track call outcomes in CRM same day.
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background:#fffbeb;border:1px solid #f59e0b;border-radius:8px;padding:16px;'>
            <h4 style='color:#b45309;'>🌡️ Warm Leads ({warm_count:,})</h4>
            <b>Priority: Nurture Campaign</b><br><br>
            • Enrol in a <b>3–5 email nurture sequence</b> over 2 weeks.<br>
            • Share relevant success stories and course testimonials.<br>
            • Invite to a <b>free webinar or live Q&A</b> session.<br>
            • Re-engage after 7 days with a follow-up call.<br>
            • Monitor for engagement signals that upgrade them to Hot.
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='background:#eff6ff;border:1px solid #3b82d4;border-radius:8px;padding:16px;'>
            <h4 style='color:#1d4ed8;'>🥶 Cold Leads ({cold_count:,})</h4>
            <b>Priority: Low-cost Re-engagement</b><br><br>
            • Add to a <b>monthly newsletter</b> drip campaign.<br>
            • Send one <b>re-engagement email</b> with a compelling offer.<br>
            • Do <b>not</b> assign to expensive outbound calling queues.<br>
            • Tag in CRM for seasonal promotions (e.g. year-end discounts).<br>
            • Review after 30 days; archive if still unresponsive.
            </div>""", unsafe_allow_html=True)

        # ── Download Buttons ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📥 Download Reports")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.download_button(
                label="⬇️ Download Cleaned Dataset (CSV)",
                data=df_to_csv_bytes(df),
                file_name="leads_cleaned.csv",
                mime="text/csv",
            )
        with col_d2:
            st.download_button(
                label="⬇️ Download Lead Priority Table (CSV)",
                data=df_to_csv_bytes(display_df),
                file_name="lead_priority.csv",
                mime="text/csv",
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("Lead Conversion Intelligence System · X Education Lead Scoring Dataset · Capstone Project")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()
