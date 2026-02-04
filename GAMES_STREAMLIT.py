import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page configuration
st.set_page_config(page_title="Game Sales Analytics Dashboard", layout="wide")

# Title and Introduction
st.title("🎮 Game Sales Business Analytics Dashboard")
st.markdown("""
This dashboard provides an interactive overview of game sales data.
**New Features:** Seasonality analysis, per-title efficiency metrics, and regional genre-fit heatmaps.
""")

# --- Data Loading & Caching ---
@st.cache_data
def load_data():
    # USE YOUR ABSOLUTE PATH HERE
    file_path = r"E:\Labmentix Internship\Game Sales\Project Files\GAME SALES DATA FINAL.csv"
    
    if not os.path.exists(file_path):
        st.error(f"File not found at: {file_path}. Please check the path.")
        st.stop()
        
    df = pd.read_csv(file_path)
    
    # Cleaning
    df_clean = df.drop_duplicates(subset=['Name', 'Platform', 'Global_Sales'])
    
    # Date Parsing for Seasonality Analysis
    df_clean['Release_Date'] = pd.to_datetime(df_clean['Release_Date'], dayfirst=True, errors='coerce')
    df_clean['Month_Name'] = df_clean['Release_Date'].dt.strftime('%b')
    df_clean['Month_Num'] = df_clean['Release_Date'].dt.month
    
    return df_clean

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

# Genre Filter
genres = ['All'] + sorted(df['Genre'].unique().tolist())
selected_genre = st.sidebar.selectbox("Select Genre", genres)

# Platform Filter
platforms = ['All'] + sorted(df['Platform'].unique().tolist())
selected_platform = st.sidebar.selectbox("Select Platform", platforms)

# Year Filter
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Apply Filters
df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

if selected_genre != 'All':
    df_filtered = df_filtered[df_filtered['Genre'] == selected_genre]

if selected_platform != 'All':
    df_filtered = df_filtered[df_filtered['Platform'] == selected_platform]

# --- KPI Section ---
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
total_global_sales = df_filtered['Global_Sales'].sum()
top_game = df_filtered.loc[df_filtered['Global_Sales'].idxmax(), 'Name'] if not df_filtered.empty else "N/A"
top_publisher = df_filtered.groupby('Publisher')['Global_Sales'].sum().idxmax() if not df_filtered.empty else "N/A"
avg_rating = df_filtered['Rating'].mean()

with col1:
    st.metric("Total Global Sales", f"${total_global_sales:,.2f}M")
with col2:
    st.metric("Top Selling Game", top_game)
with col3:
    st.metric("Top Publisher", top_publisher)
with col4:
    st.metric("Avg User Rating", f"{avg_rating:.2f}" if not df_filtered.empty else "N/A")

st.markdown("---")

# --- Visualizations ---

# Row 1: Sales Trends and Regional Breakdown
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Global Sales Trend Over Time")
    sales_by_year = df_filtered.groupby('Year')['Global_Sales'].sum().reset_index()
    fig_line = px.line(sales_by_year, x='Year', y='Global_Sales', markers=True, 
                       title="Total Sales by Year", template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

with col_right:
    st.subheader("Regional Sales Distribution")
    region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    sales_by_region = df_filtered[region_cols].sum().reset_index()
    sales_by_region.columns = ['Region', 'Sales']
    
    fig_pie = px.pie(sales_by_region, values='Sales', names='Region', 
                     title="Sales Share by Region", hole=0.4, template="plotly_dark")
    st.plotly_chart(fig_pie, use_container_width=True)

# Row 2: Genre and Platform Analysis
col_left_2, col_right_2 = st.columns(2)

with col_left_2:
    st.subheader("Top Genres by Total Sales")
    sales_by_genre = df_filtered.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).reset_index()
    fig_bar_genre = px.bar(sales_by_genre, x='Global_Sales', y='Genre', orientation='h', 
                           title="Global Sales by Genre", template="plotly_dark", color='Global_Sales')
    st.plotly_chart(fig_bar_genre, use_container_width=True)

with col_right_2:
    st.subheader("Top Platforms by Sales")
    sales_by_platform = df_filtered.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10).reset_index()
    fig_bar_plat = px.bar(sales_by_platform, x='Platform', y='Global_Sales', 
                          title="Top 10 Platforms by Sales", template="plotly_dark", color='Global_Sales')
    st.plotly_chart(fig_bar_plat, use_container_width=True)

st.markdown("---")
st.header("Strategic Business Analysis")

# Row 3: Seasonality & Efficiency
col_season, col_eff = st.columns(2)

with col_season:
    st.subheader("Launch Window Analysis (Seasonality)")
    # Group by Month Number first to sort correctly, then map to Name
    if 'Month_Num' in df_filtered.columns:
        seasonal_sales = df_filtered.groupby('Month_Num')['Global_Sales'].sum().reset_index()
        # Map numbers to names for display
        import calendar
        seasonal_sales['Month'] = seasonal_sales['Month_Num'].apply(lambda x: calendar.month_abbr[int(x)] if pd.notnull(x) else 'Unknown')
        
        fig_season = px.bar(seasonal_sales, x='Month', y='Global_Sales',
                            title="Total Sales by Release Month", template="plotly_dark",
                            color='Global_Sales', color_continuous_scale='Oranges')
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.write("Date data unavailable for filtered selection.")

with col_eff:
    st.subheader("Genre Efficiency: Avg Revenue per Title")
    # Calculate Average Sales per Game for each Genre
    genre_efficiency = df_filtered.groupby('Genre').agg({'Global_Sales':'sum', 'Name':'count'}).reset_index()
    genre_efficiency['Avg_Sales_Per_Game'] = genre_efficiency['Global_Sales'] / genre_efficiency['Name']
    genre_efficiency = genre_efficiency.sort_values(by='Avg_Sales_Per_Game', ascending=False)
    
    fig_eff = px.bar(genre_efficiency, x='Genre', y='Avg_Sales_Per_Game',
                     title="Average Revenue per Game Title (Yield)", template="plotly_dark",
                     color='Avg_Sales_Per_Game', color_continuous_scale='Tealgrn')
    st.plotly_chart(fig_eff, use_container_width=True)

# Row 4: Advanced Market View (Treemap & Heatmap)
col_tree, col_heat = st.columns(2)

with col_tree:
    st.subheader("Market Hierarchy (Publisher > Genre)")
    # Taking top 20 publishers to keep the Treemap readable
    top_publishers = df_filtered.groupby('Publisher')['Global_Sales'].sum().nlargest(20).index
    df_tree = df_filtered[df_filtered['Publisher'].isin(top_publishers)]
    
    fig_tree = px.treemap(df_tree, path=['Publisher', 'Genre'], values='Global_Sales',
                          title="Market Share by Publisher & Genre (Top 20)", template="plotly_dark")
    st.plotly_chart(fig_tree, use_container_width=True)

with col_heat:
    st.subheader("Regional Product Fit (Heatmap)")
    # Pivot data for heatmap: Genre vs Region
    region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    genre_region = df_filtered.groupby('Genre')[region_cols].sum().reset_index()
    # Melt for heatmap format
    genre_region_melt = genre_region.melt(id_vars='Genre', var_name='Region', value_name='Sales')
    
    fig_heat = px.density_heatmap(genre_region_melt, x='Region', y='Genre', z='Sales',
                                  title="Sales Intensity: Genre vs Region", template="plotly_dark",
                                  color_continuous_scale='Viridis')
    st.plotly_chart(fig_heat, use_container_width=True)

# Row 5: Publisher Trends & Engagement (Original Advanced Charts)
col_pub, col_corr = st.columns(2)

with col_pub:
    st.subheader("Publisher Performance")
    publisher_sales = df_filtered.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10).reset_index()
    fig_pub = px.bar(publisher_sales, x='Global_Sales', y='Publisher', orientation='h',
                     title="Top 10 Publishers by Sales", template="plotly_dark", color='Global_Sales')
    st.plotly_chart(fig_pub, use_container_width=True)

with col_corr:
    st.subheader("Metric Correlations")
    corr_cols = ['Global_Sales', 'Rating', 'Plays', 'Playing', 'Backlogs', 'Wishlist', 'Number_of_Reviews']
    if not df_filtered.empty:
        corr_matrix = df_filtered[corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                             title="Correlation Matrix", template="plotly_dark", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("Not enough data.")

# --- Raw Data View ---
with st.expander("View Raw Data"):
    st.dataframe(df_filtered)