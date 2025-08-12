# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Page config & style
# -------------------------
st.set_page_config(page_title="Housing Price Dashboard", layout="wide", initial_sidebar_state="expanded")
sns.set(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams["figure.figsize"] = (8, 5)

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_data(path="Housing.csv"):
    df = pd.read_csv(path)
    # normalize column names (optional)
    df.columns = df.columns.str.strip()
    return df

def currency(x):
    try:
        return f"₹{int(x):,}"
    except Exception:
        return x

def safe_show_empty(df_filtered):
    if df_filtered.empty:
        st.warning("No rows match the selected filters. Please change the filters to see visuals.")
        return True
    return False

# -------------------------
# Load data
# -------------------------
st.title("🏠 Housing Price Analysis Dashboard")
st.markdown(
    "Interactive dashboard to explore the Housing dataset — filtering, visuals, and quick insights. "
    "Data source: `Housing.csv`."
)

df = load_data()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("🔎 Filters")

# Prepare filter options safely
furnish_opts = sorted(df['furnishingstatus'].unique()) if 'furnishingstatus' in df.columns else []
bedroom_opts = sorted(df['bedrooms'].unique()) if 'bedrooms' in df.columns else []
stories_opts = sorted(df['stories'].unique()) if 'stories' in df.columns else []
bathroom_opts = sorted(df['bathrooms'].unique()) if 'bathrooms' in df.columns else []

furnishing_filter = st.sidebar.multiselect("Furnishing Status", options=furnish_opts, default=furnish_opts)
bedroom_filter = st.sidebar.multiselect("Bedrooms", options=bedroom_opts, default=bedroom_opts)
stories_filter = st.sidebar.multiselect("Stories", options=stories_opts, default=stories_opts)
bathroom_filter = st.sidebar.multiselect("Bathrooms", options=bathroom_opts, default=bathroom_opts)

# Additional toggles
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
show_pairplot = st.sidebar.checkbox("Show Pairplot (may be slow)", value=False)

# Apply filters (robust)
df_filtered = df.copy()
try:
    if furnish_opts:
        df_filtered = df_filtered[df_filtered['furnishingstatus'].isin(furnishing_filter)]
    if bedroom_opts:
        df_filtered = df_filtered[df_filtered['bedrooms'].isin(bedroom_filter)]
    if stories_opts:
        df_filtered = df_filtered[df_filtered['stories'].isin(stories_filter)]
    if bathroom_opts:
        df_filtered = df_filtered[df_filtered['bathrooms'].isin(bathroom_filter)]
except KeyError:
    # in case some columns are missing, ignore and continue
    pass

# -------------------------
# Top: Data preview & summary
# -------------------------
st.subheader("📋 Data Preview")
st.dataframe(df_filtered.head(10))

st.subheader("📊 Summary Statistics")
st.write(df_filtered.describe(include='all').T)

if safe_show_empty(df_filtered):
    st.stop()  # Stop further rendering if no data after filters

# -------------------------
# Row 1: Scatter (price vs area) and Avg price bars
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💰 Price vs Area")
    fig1, ax1 = plt.subplots()
    hue_col = 'furnishingstatus' if 'furnishingstatus' in df_filtered.columns else None
    sns.scatterplot(data=df_filtered, x='area', y='price', hue=hue_col, alpha=0.7, ax=ax1, legend='brief')
    ax1.set_xlabel("Area (sq ft)")
    ax1.set_ylabel("Price")
    if hue_col:
        ax1.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    st.pyplot(fig1)

with col2:
    st.markdown("### 🏷️ Average Price by Furnishing Status")
    if 'furnishingstatus' in df_filtered.columns:
        avg_price = df_filtered.groupby('furnishingstatus')['price'].mean().sort_values()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=avg_price.values, y=avg_price.index, palette="coolwarm", ax=ax2)
        ax2.set_xlabel("Average Price")
        ax2.set_ylabel("Furnishing Status")
        for i, v in enumerate(avg_price.values):
            ax2.text(v, i, f" {int(v):,}", va='center')
        st.pyplot(fig2)
    else:
        st.info("`furnishingstatus` column not found.")

# -------------------------
# Row 2: Heatmap + Countplot
# -------------------------
col3, col4 = st.columns(2)

with col3:
    if show_heatmap:
        st.markdown("### 📈 Correlation Heatmap (numeric)")
        numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numeric_cols) >= 2:
            fig3, ax3 = plt.subplots(figsize=(7, 6))
            corr = df_filtered[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax3)
            st.pyplot(fig3)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")
    else:
        st.info("Heatmap is hidden. Toggle 'Show Correlation Heatmap' in the sidebar to enable it.")

with col4:
    st.markdown("### 🛏️ Bedroom Count Distribution")
    if 'bedrooms' in df_filtered.columns:
        fig4, ax4 = plt.subplots()
        sns.countplot(x='bedrooms', data=df_filtered, palette="Set2", ax=ax4)
        st.pyplot(fig4)
    else:
        st.info("`bedrooms` column not found.")

# -------------------------
# Row 3: Distribution & outliers
# -------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown("### 📉 Price Distribution")
    fig5, ax5 = plt.subplots()
    sns.histplot(df_filtered['price'], bins=30, kde=True, ax=ax5)
    ax5.set_xlabel("Price")
    st.pyplot(fig5)

with col6:
    st.markdown("### 📦 Area Boxplot (outliers)")
    fig6, ax6 = plt.subplots()
    sns.boxplot(x=df_filtered['area'], color='skyblue', ax=ax6)
    st.pyplot(fig6)

# -------------------------
# Row 4: Bathrooms & Parking impact
# -------------------------
col7, col8 = st.columns(2)

with col7:
    st.markdown("### 🚿 Average Price by Bathrooms")
    if 'bathrooms' in df_filtered.columns:
        fig7, ax7 = plt.subplots()
        sns.barplot(x='bathrooms', y='price', data=df_filtered, estimator=np.mean, palette="viridis", ax=ax7)
        ax7.set_ylabel("Average Price")
        st.pyplot(fig7)
    else:
        st.info("`bathrooms` column not found.")

with col8:
    st.markdown("### 🚗 Parking vs Price (boxplot)")
    if 'parking' in df_filtered.columns:
        fig8, ax8 = plt.subplots()
        sns.boxplot(x='parking', y='price', data=df_filtered, palette="Set3", ax=ax8)
        st.pyplot(fig8)
    else:
        st.info("`parking` column not found.")

# -------------------------
# Row 5: Stories & Furnishing pie
# -------------------------
st.markdown("### 🏠 Average Price by Stories")
if 'stories' in df_filtered.columns:
    fig9, ax9 = plt.subplots()
    sns.barplot(x='stories', y='price', data=df_filtered, estimator=np.mean, palette="coolwarm", ax=ax9)
    ax9.set_ylabel("Average Price")
    st.pyplot(fig9)
else:
    st.info("`stories` column not found.")

st.markdown("### 🛋️ Furnishing Status Distribution")
if 'furnishingstatus' in df_filtered.columns:
    furn_counts = df_filtered['furnishingstatus'].value_counts()
    fig10, ax10 = plt.subplots()
    ax10.pie(furn_counts.values, labels=furn_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax10.axis('equal')
    st.pyplot(fig10)
else:
    st.info("`furnishingstatus` column not found for pie chart.")

# -------------------------
# Optional pairplot (slow)
# -------------------------
if show_pairplot:
    st.markdown("### 🔗 Pairplot (selected numeric features)")
    numeric_cols_small = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()[:6]
    if len(numeric_cols_small) >= 2:
        pair_df = df_filtered[numeric_cols_small]
        # seaborn.pairplot returns its own figure; use st.pyplot on that fig
        pp = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha":0.5})
        st.pyplot(pp.fig)
        plt.close('all')
    else:
        st.info("Not enough numeric columns for pairplot.")

# -------------------------
# Correlation with price (printed)
# -------------------------
st.markdown("### 🔬 Correlation of Price with Numeric Features")
num_cols_for_corr = [c for c in ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking'] if c in df_filtered.columns]
if len(num_cols_for_corr) >= 2:
    corr_with_price = df_filtered[num_cols_for_corr].corr()['price'].sort_values(ascending=False)
    st.write(corr_with_price)
else:
    st.info("Not enough numeric columns to compute price correlations.")

# -------------------------
# Key Insights panel
# -------------------------
st.subheader("📌 Key Insights (Dynamic)")
try:
    total = len(df_filtered)
    avg_price_f = df_filtered['price'].mean()
    largest_area = df_filtered['area'].max()
    most_common_furnish = df_filtered['furnishingstatus'].mode()[0] if 'furnishingstatus' in df_filtered.columns else "N/A"

    st.markdown(f"""
    - **Total properties shown:** {total}
    - **Average price (filtered):** {currency(avg_price_f)}
    - **Largest area (filtered):** {largest_area:,} sq ft
    - **Most common furnishing status:** {most_common_furnish}
    """)
except Exception as e:
    st.write("Could not compute some insights:", e)

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.markdown("✅ Built with Streamlit • Use the sidebar to update filters • Dataset: `Housing.csv`")
