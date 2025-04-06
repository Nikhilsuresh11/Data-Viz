import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import io
import uuid
import re
from collections import Counter
import base64
from datetime import datetime
import tempfile
import requests
from bs4 import BeautifulSoup
try:
    import tabula
    import PyPDF2
    import docx
except ImportError:
    st.warning("Some document processing libraries are not installed. PDF and DOC processing may not work.")

# Set page configuration
st.set_page_config(
    page_title="Data Viz App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state if not already done
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'correlations' not in st.session_state:
    st.session_state.correlations = None
if 'potential_targets' not in st.session_state:
    st.session_state.potential_targets = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'chart_ideas' not in st.session_state:
    st.session_state.chart_ideas = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Upload"

# Helper functions for data analysis
def analyze_column_types(df):
    """Identify column types (numeric, categorical, datetime, text)"""
    column_types = {}
    
    for col in df.columns:
        # Check if datetime
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.to_datetime(df[col], errors='coerce').notna().all():
                column_types[col] = "datetime"
                continue
        except:
            pass
            
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if binary (0/1)
            if set(df[col].dropna().unique()).issubset({0, 1}):
                column_types[col] = "binary"
            # Check if likely ID column
            elif ("id" in col.lower() or "key" in col.lower()) and df[col].nunique() > 0.9 * len(df):
                column_types[col] = "id"
            else:
                column_types[col] = "numeric"
        # Check if categorical
        elif df[col].nunique() < min(0.2 * len(df), 50):
            column_types[col] = "categorical"
        # Must be text
        else:
            column_types[col] = "text"
            
    return column_types

def get_column_stats(df):
    """Generate basic statistics for each column"""
    stats = {}
    
    for col in df.columns:
        col_stats = {
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": float((df[col].isna().sum() / len(df)) * 100),
            "unique_count": int(df[col].nunique()),
            "unique_pct": float((df[col].nunique() / len(df)) * 100)
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "skew": float(df[col].skew()) if len(df) > 3 and not pd.isna(df[col].skew()) else None
            })
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 30:
            top_values = df[col].value_counts().head(5).to_dict()
            col_stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
        stats[col] = col_stats
        
    return stats

def identify_correlations(df, column_types):
    """Find correlations between numeric columns"""
    numeric_cols = [col for col, type_ in column_types.items() 
                   if type_ in ["numeric", "binary"] and col in df.columns]
    
    if len(numeric_cols) < 2:
        return {}
    
    correlations = {}
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Get the pairs with highest correlation
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    sorted_corrs = upper_tri.stack().sort_values(ascending=False)
    
    for (col1, col2), corr_value in sorted_corrs.items():
        if corr_value > 0.5:  # Only consider meaningful correlations
            correlations[(col1, col2)] = float(corr_value)
            
    return correlations

def identify_potential_targets(df, column_types):
    """Identify columns that might be prediction targets"""
    potential_targets = []
    
    # Binary or categorical columns with few unique values often make good targets
    for col, type_ in column_types.items():
        if type_ in ["binary", "categorical"] and df[col].nunique() < 10:
            potential_targets.append(col)
    
    # Numeric columns that aren't IDs and have meaningful variation
    for col, type_ in column_types.items():
        if type_ == "numeric" and "id" not in col.lower() and df[col].std() > 0:
            potential_targets.append(col)
            
    return potential_targets

def generate_insights(df, column_types, stats, correlations, potential_targets):
    """Generate basic insights about the data"""
    insights = []
    
    # Dataset overview
    insights.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
    
    # Column types summary
    col_type_counts = Counter(column_types.values())
    insights.append(f"Contains {col_type_counts.get('numeric', 0)} numeric, {col_type_counts.get('categorical', 0)} categorical, {col_type_counts.get('datetime', 0)} datetime columns.")
    
    # Missing values
    missing_cols = {col: stats[col]["missing_pct"] for col in df.columns if stats[col]["missing_pct"] > 0}
    if missing_cols:
        insights.append(f"{len(missing_cols)} columns have missing values. Highest: {max(missing_cols.items(), key=lambda x: x[1])[0]} ({max(missing_cols.values()):.1f}%).")
    else:
        insights.append("No missing values found in the dataset.")
    
    # Correlations
    if correlations:
        top_corr = max(correlations.items(), key=lambda x: x[1])
        insights.append(f"Strongest correlation: {top_corr[0][0]} and {top_corr[0][1]} ({top_corr[1]:.2f}).")
    
    # Potential targets
    if potential_targets:
        insights.append(f"Potential target variables: {', '.join(potential_targets[:3])}.")
    
    # Unusual distributions
    skewed_cols = []
    for col, col_stats in stats.items():
        if col_stats.get("skew") and abs(col_stats["skew"]) > 2:
            skewed_cols.append(col)
    
    if skewed_cols:
        insights.append(f"Columns with highly skewed distributions: {', '.join(skewed_cols[:3])}.")
    
    return insights

def generate_visualization_recommendations(df, column_types, stats, correlations, potential_targets):
    """Generate visualization recommendations based on data analysis"""
    recommendations = []
    
    # 1. Distribution of numeric features
    numeric_cols = [col for col, type_ in column_types.items() if type_ == "numeric" and stats[col]["missing_pct"] < 50]
    if numeric_cols:
        recommendations.append({
            "type": "distribution",
            "title": "Distribution of Numeric Features",
            "columns": numeric_cols[:min(5, len(numeric_cols))],
            "description": "Histogram or KDE plots to understand data distributions"
        })
    
    # 2. Categorical feature counts
    cat_cols = [col for col, type_ in column_types.items() if type_ == "categorical" and stats[col]["missing_pct"] < 50]
    if cat_cols:
        recommendations.append({
            "type": "categorical_counts",
            "title": "Categorical Feature Counts",
            "columns": cat_cols[:min(5, len(cat_cols))],
            "description": "Bar charts showing frequency of each category"
        })
    
    # 3. Time series if datetime columns exist
    time_cols = [col for col, type_ in column_types.items() if type_ == "datetime"]
    if time_cols and numeric_cols:
        recommendations.append({
            "type": "time_series",
            "title": "Time Series Analysis",
            "time_column": time_cols[0],
            "value_column": numeric_cols[0],
            "description": "Visualize trends over time"
        })
    
    # 4. Correlation heatmap for numeric features
    if len(numeric_cols) > 2:
        recommendations.append({
            "type": "correlation",
            "title": "Correlation Matrix",
            "columns": numeric_cols[:min(10, len(numeric_cols))],
            "description": "Identify relationships between numeric variables"
        })
    
    # 5. Scatter plots for highly correlated features
    top_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
    for (col1, col2), corr_value in top_correlations:
        recommendations.append({
            "type": "scatter",
            "title": f"Relationship: {col1} vs {col2}",
            "x": col1,
            "y": col2,
            "description": f"Strong correlation ({corr_value:.2f}) between these variables"
        })
    
    # 6. Box plots for categorical vs numeric
    if cat_cols and numeric_cols:
        for cat_col in cat_cols[:min(2, len(cat_cols))]:
            if df[cat_col].nunique() <= 10:  # Limit to avoid overcrowded plots
                for num_col in numeric_cols[:min(2, len(numeric_cols))]:
                    recommendations.append({
                        "type": "boxplot",
                        "title": f"{num_col} by {cat_col}",
                        "cat_column": cat_col,
                        "num_column": num_col,
                        "description": "Compare distributions across categories"
                    })
    
    # 7. Missing value visualization if significant
    cols_with_missing = [col for col in df.columns if stats[col]["missing_pct"] > 5]
    if cols_with_missing:
        recommendations.append({
            "type": "missing_values",
            "title": "Missing Value Analysis",
            "columns": cols_with_missing,
            "description": "Visualize patterns in missing data"
        })
    
    # 8. Pair plot for potential multi-variable relationships
    if len(numeric_cols) >= 3 and len(numeric_cols) <= 6:
        recommendations.append({
            "type": "pairplot",
            "title": "Multi-variable Relationships",
            "columns": numeric_cols[:4],  # Limit to avoid overcrowding
            "description": "Explore relationships between multiple variables"
        })
    
    return recommendations

def plot_distributions(df, columns):
    """Plot distributions of numeric columns"""
    fig = go.Figure()
    for col in columns:
        try:
            # Filter out non-numeric data
            numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
            
            # Generate histogram
            fig.add_trace(go.Histogram(
                x=numeric_data,
                name=col,
                opacity=0.7,
                histnorm='probability density'
            ))
        except Exception as e:
            st.error(f"Couldn't plot distribution for {col}: {str(e)}")
            
    fig.update_layout(
        title="Distribution of Numeric Features",
        xaxis_title="Value",
        yaxis_title="Density",
        barmode='overlay',
        height=500
    )
    return fig

def plot_categorical_counts(df, columns):
    """Plot categorical counts"""
    if len(columns) == 1:
        # Single column bar chart
        col = columns[0]
        counts = df[col].value_counts().nlargest(20)  # Limit to top 20 categories
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            labels={'x': col, 'y': 'Count'},
            title=f"Distribution of {col}",
            height=500
        )
    else:
        # Subplots for multiple columns
        fig = make_subplots(rows=len(columns), cols=1, subplot_titles=[f"Distribution of {col}" for col in columns])
        
        for i, col in enumerate(columns):
            counts = df[col].value_counts().nlargest(10)  # Limit to top 10 categories
            fig.add_trace(
                go.Bar(x=counts.index, y=counts.values, name=col),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(columns),
            showlegend=False
        )
    
    return fig

def plot_correlation_matrix(df, columns):
    """Plot correlation matrix for numeric columns"""
    # Calculate correlation
    corr_df = df[columns].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Correlation Matrix"
    )
    
    fig.update_layout(height=600)
    return fig

def plot_scatter(df, x_col, y_col):
    """Create scatter plot for two numeric columns"""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=f"{x_col} vs {y_col}",
        opacity=0.7,
        height=500
    )
    
    return fig

def plot_boxplot(df, cat_col, num_col):
    """Create box plot for categorical vs numeric"""
    # Limit to top 10 categories if there are many
    if df[cat_col].nunique() > 10:
        top_cats = df[cat_col].value_counts().nlargest(10).index.tolist()
        df_plot = df[df[cat_col].isin(top_cats)].copy()
        title = f"Top 10 Categories: {num_col} by {cat_col}"
    else:
        df_plot = df.copy()
        title = f"{num_col} by {cat_col}"
        
    fig = px.box(
        df_plot, 
        x=cat_col, 
        y=num_col,
        title=title,
        height=500
    )
    
    return fig

def plot_missing_values(df, columns):
    """Create visualization of missing values"""
    # Prepare data for visualization
    missing_data = pd.DataFrame({
        'Column': columns,
        'Missing': [df[col].isna().sum() for col in columns],
        'Percentage': [df[col].isna().sum() / len(df) * 100 for col in columns]
    }).sort_values('Percentage', ascending=False)
    
    fig = px.bar(
        missing_data,
        x='Column',
        y='Percentage',
        title="Missing Values by Column (%)",
        height=500
    )
    
    return fig

def plot_time_series(df, time_col, value_col):
    """Create time series plot"""
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Sort by time
    df_sorted = df.sort_values(by=time_col)
    
    fig = px.line(
        df_sorted,
        x=time_col,
        y=value_col,
        title=f"{value_col} over time",
        height=500
    )
    
    return fig

def plot_pairplot(df, columns):
    """Create pairplot matrix"""
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        title="Multi-variable Relationships",
        opacity=0.7
    )
    
    fig.update_layout(height=700)
    return fig

# Main app layout
st.title("📊 Data Visualization App")

# Define tabs
tabs = ["Upload", "Overview", "Explorer", "Visualizations", "Insights", "Custom"]
tab_icons = ["📤", "📋", "🔍", "📈", "💡", "🎨"]

# Create horizontal tabs
cols = st.columns(len(tabs))
for i, (col, tab, icon) in enumerate(zip(cols, tabs, tab_icons)):
    if col.button(f"{icon} {tab}", key=f"tab_{tab}", use_container_width=True, 
                 disabled=st.session_state.dataframe is None and tab != "Upload"):
        st.session_state.current_tab = tab

# Render selected tab
st.write("---")
if st.session_state.current_tab == "Upload":
    st.header("📤 Upload Data")
    
    upload_method = st.radio("Select upload method:", 
                            ["Upload file", "From URL", "Sample data"],
                            horizontal=True)
    
    if upload_method == "Upload file":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"File uploaded successfully! Detected {len(df)} rows and {len(df.columns)} columns.")
                
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Process the data
                if st.button("Process Data", key="process_upload"):
                    with st.spinner("Analyzing data..."):
                        # Store the dataframe
                        st.session_state.dataframe = df
                        
                        # Analyze the data
                        column_types = analyze_column_types(df)
                        stats = get_column_stats(df) 
                        correlations = identify_correlations(df, column_types)
                        potential_targets = identify_potential_targets(df, column_types)
                        
                        # Generate visualization recommendations
                        recommendations = generate_visualization_recommendations(df, column_types, stats, correlations, potential_targets)
                        
                        # Generate insights
                        insights = generate_insights(df, column_types, stats, correlations, potential_targets)
                        
                        # Store analysis results in session state
                        st.session_state.column_types = column_types
                        st.session_state.stats = stats
                        st.session_state.correlations = correlations
                        st.session_state.potential_targets = potential_targets
                        st.session_state.recommendations = recommendations
                        st.session_state.insights = insights
                        
                        # Switch tab to Overview
                        st.session_state.current_tab = "Overview"
                        st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif upload_method == "From URL":
        url = st.text_input("Enter URL of CSV or Excel file:")
        
        if url and st.button("Download and Process", key="url_download"):
            try:
                with st.spinner("Downloading file..."):
                    # Get file from URL
                    response = requests.get(url)
                    if response.status_code == 200:
                        # Determine file type from URL
                        if url.endswith('.csv'):
                            df = pd.read_csv(io.StringIO(response.text))
                        elif url.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(io.BytesIO(response.content))
                        else:
                            st.error("URL must point to a CSV or Excel file")
                            st.stop()
                        
                        st.success(f"File downloaded successfully! Detected {len(df)} rows and {len(df.columns)} columns.")
                        
                        with st.expander("Data Preview", expanded=True):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Process button
                        if st.button("Process Data", key="process_url"):
                            process_data(df)
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")
    
    elif upload_method == "Sample data":
        sample_dataset = st.selectbox(
            "Choose a sample dataset:",
            ["Iris", "Titanic", "Boston Housing", "Wine Quality", "Diabetes"]
        )
        
        if st.button("Load Sample Data", key="load_sample"):
            with st.spinner("Loading sample data..."):
                try:
                    if sample_dataset == "Iris":
                        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
                    elif sample_dataset == "Titanic":
                        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
                    elif sample_dataset == "Boston Housing":
                        from sklearn.datasets import load_boston
                        boston = load_boston()
                        df = pd.DataFrame(boston.data, columns=boston.feature_names)
                        df['PRICE'] = boston.target
                    elif sample_dataset == "Wine Quality":
                        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
                    elif sample_dataset == "Diabetes":
                        from sklearn.datasets import load_diabetes
                        diabetes = load_diabetes()
                        df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
                        df['target'] = diabetes.target
                    
                    st.success(f"Sample data loaded! Detected {len(df)} rows and {len(df.columns)} columns.")
                    
                    with st.expander("Data Preview", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Process the data
                    if st.button("Process Data", key="process_sample"):
                        process_data(df)
                
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")

# Function to process dataframe
def process_data(df):
    with st.spinner("Analyzing data..."):
        # Store the dataframe
        st.session_state.dataframe = df
        
        # Analyze the data
        column_types = analyze_column_types(df)
        stats = get_column_stats(df) 
        correlations = identify_correlations(df, column_types)
        potential_targets = identify_potential_targets(df, column_types)
        
        # Generate visualization recommendations
        recommendations = generate_visualization_recommendations(df, column_types, stats, correlations, potential_targets)
        
        # Generate insights
        insights = generate_insights(df, column_types, stats, correlations, potential_targets)
        
        # Store analysis results in session state
        st.session_state.column_types = column_types
        st.session_state.stats = stats
        st.session_state.correlations = correlations
        st.session_state.potential_targets = potential_targets
        st.session_state.recommendations = recommendations
        st.session_state.insights = insights
        
        # Switch tab to Overview
        st.session_state.current_tab = "Overview"
        st.experimental_rerun()

if st.session_state.current_tab == "Overview":
    st.header("📋 Data Overview")
    
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        column_types = st.session_state.column_types
        stats = st.session_state.stats
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        
        # Calculate missing values percentage
        missing_pct = sum(stats[col]['missing_count'] for col in stats) / (len(df) * len(df.columns)) * 100
        col3.metric("Missing Values", f"{missing_pct:.1f}%")
        
        # Column type counts
        col_type_counts = Counter(column_types.values())
        
        st.subheader("Column Types")
        cols = st.columns(len(col_type_counts))
        for i, (col_type, count) in enumerate(col_type_counts.items()):
            cols[i].metric(col_type.capitalize(), count)
        
        # Missing values by column
        st.subheader("Columns with Missing Values")
        missing_cols = {col: stats[col]["missing_pct"] for col in df.columns if stats[col]["missing_pct"] > 0}
        if missing_cols:
            missing_df = pd.DataFrame({
                "Column": list(missing_cols.keys()),
                "Missing (%)": list(missing_cols.values())
            }).sort_values("Missing (%)", ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.info("No missing values found in the dataset.")
        
        # Sample data
        st.subheader("Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Key insights
        st.subheader("Quick Insights")
        for insight in st.session_state.insights[:5]:  # Show top 5 insights
            st.markdown(f"• {insight}")
    else:
        st.info("Please upload a dataset in the Upload tab first.")

elif st.session_state.current_tab == "Explorer":
    st.header("🔍 Data Explorer")
    
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        column_types = st.session_state.column_types
        stats = st.session_state.stats
        
        # Column selection and filtering
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Column Filter")
            
            # Filter by column type
            col_type_filter = st.multiselect(
                "Column Type",
                options=list(set(column_types.values())),
                default=list(set(column_types.values()))
            )
            
            # Filter columns by search
            search_term = st.text_input("Search Columns", "")
            
            # Get filtered columns
            filtered_columns = [col for col, type_ in column_types.items() 
                              if type_ in col_type_filter and 
                              (search_term.lower() in col.lower() or not search_term)]
            
            # Column selector
            selected_column = st.selectbox("Select Column", filtered_columns)
            
        with col2:
            if selected_column:
                st.subheader(f"Column: {selected_column}")
                
                # Column type and stats
                col_type = column_types[selected_column]
                col_stats = stats[selected_column]
                
                # Display key statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Type", col_type.capitalize())
                col2.metric("Unique Values", col_stats["unique_count"])
                col3.metric("Missing", f"{col_stats['missing_pct']:.1f}%")
                
                if col_type == "numeric":
                    col4.metric("Mean", f"{col_stats['mean']:.2f}")
                    
                    # Show detailed numeric stats
                    st.subheader("Statistics")
                    stat_cols = st.columns(6)
                    stat_cols[0].metric("Min", f"{col_stats['min']:.2f}")
                    stat_cols[1].metric("25%", f"{df[selected_column].quantile(0.25):.2f}")
                    stat_cols[2].metric("Median", f"{col_stats['median']:.2f}")
                    stat_cols[3].metric("75%", f"{df[selected_column].quantile(0.75):.2f}")
                    stat_cols[4].metric("Max", f"{col_stats['max']:.2f}")
                    stat_cols[5].metric("Std Dev", f"{col_stats['std']:.2f}")
                    
                    # Distribution plot
                    st.subheader("Distribution")
                    fig = px.histogram(
                        df, 
                        x=selected_column,
                        marginal="box",
                        histnorm="probability density",
                        title=f"Distribution of {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif col_type == "categorical" or col_type == "binary":
                    # Show value counts
                    st.subheader("Value Counts")
                    value_counts = df[selected_column].value_counts().reset_index()
                    value_counts.columns = [selected_column, "Count"]
                    
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.dataframe(value_counts, use_container_width=True)
                    
                    with col2:
                        # Bar chart of value counts
                        fig = px.bar(
                            value_counts.head(20),  # Limit to top 20
                            x=selected_column,
                            y="Count",
                            title=f"Top Values for {selected_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif col_type == "datetime":
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df[selected_column]):
                        date_series = pd.to_datetime(df[selected_column], errors='coerce')
                    else:
                        date_series = df[selected_column]
                    
                    # Show date range
                    min_date = date_series.min()
                    max_date = date_series.max()
                    
                    col4.metric("Date Range", f"{max_date - min_date}")
                    
                    # Time distribution
                    st.subheader("Time Distribution")
                    
                    # Group by year-month
                    date_counts = date_series.dt.to_period('M').value_counts().sort_index()
                    date_df = pd.DataFrame({
                        'Month': [str(date) for date in date_counts.index],
                        'Count': date_counts.values
                    })
                    
                    fig = px.line(
                        date_df,
                        x='Month',
                        y='Count',
                        title=f"Distribution over time for {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show sample values
                st.subheader("Sample Values")
                st.dataframe(df[[selected_column]].dropna().head(10), use_container_width=True)
                
        # Raw data with filters
        st.subheader("Filtered Data View")
        
        # Add filters
        with st.expander("Add Filters"):
            filter_columns = st.multiselect("Select columns to filter", list(df.columns))
            
            filters = {}
            for col in filter_columns:
                col_type = column_types[col]
                
                if col_type == "numeric":
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    
                    filters[col] = st.slider(
                        f"Filter {col}",
                        min_val,
                        max_val,
                        (min_val, max_val)
                    )
                
                elif col_type == "categorical" or col_type == "binary":
                    unique_values = df[col].dropna().unique()
                    filters[col] = st.multiselect(
                        f"Filter {col}",
                        options=unique_values,
                        default=unique_values
                    )
                
                elif col_type == "datetime":
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        date_series = pd.to_datetime(df[col], errors='coerce')
                    else:
                        date_series = df[col]
                    
                    min_date = date_series.min().date()
                    max_date = date_series.max().date()
                    
                    filters[col] = st.date_input(
                        f"Filter {col}",
                        value=(min_date, max_date)
                    )
        
        # Apply filters to the dataframe
        filtered_df = df.copy()
        for col, filter_val in filters.items():
            col_type = column_types[col]
            
            if col_type == "numeric":
                filtered_df = filtered_df[(filtered_df[col] >= filter_val[0]) & 
                                         (filtered_df[col] <= filter_val[1])]
            
            elif col_type == "categorical" or col_type == "binary":
                if filter_val:  # Only filter if values are selected
                    filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
            
            elif col_type == "datetime":
                if len(filter_val) == 2:
                    if not pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
                        filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce')
                    
                    start_date, end_date = filter_val
                    filtered_df = filtered_df[(filtered_df[col].dt.date >= start_date) & 
                                             (filtered_df[col].dt.date <= end_date)]
        
        # Show the filtered dataframe
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button for filtered data
        if len(filters) > 0:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload a dataset in the Upload tab first.")

elif st.session_state.current_tab == "Visualizations":
    st.header("📈 Data Visualizations")
    
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        recommendations = st.session_state.recommendations
        
        # Visualization recommendations
        st.subheader("Recommended Visualizations")
        
        if recommendations:
            # Create tabs for visualization types
            viz_tabs = st.tabs([rec['title'] for rec in recommendations])
            
            for i, (tab, rec) in enumerate(zip(viz_tabs, recommendations)):
                with tab:
                    viz_type = rec['type']
                    
                    if viz_type == "distribution" and 'columns' in rec:
                        fig = plot_distributions(df, rec['columns'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "categorical_counts" and 'columns' in rec:
                        fig = plot_categorical_counts(df, rec['columns'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "correlation" and 'columns' in rec:
                        fig = plot_correlation_matrix(df, rec['columns'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "scatter" and 'x' in rec and 'y' in rec:
                        fig = plot_scatter(df, rec['x'], rec['y'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "boxplot" and 'cat_column' in rec and 'num_column' in rec:
                        fig = plot_boxplot(df, rec['cat_column'], rec['num_column'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "missing_values" and 'columns' in rec:
                        fig = plot_missing_values(df, rec['columns'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "time_series" and 'time_column' in rec and 'value_column' in rec:
                        fig = plot_time_series(df, rec['time_column'], rec['value_column'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif viz_type == "pairplot" and 'columns' in rec:
                        fig = plot_pairplot(df, rec['columns'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show description
                    if 'description' in rec:
                        st.info(rec['description'])
        else:
            st.info("No visualization recommendations available.")
        
        # Custom visualization builder
        st.subheader("Build Your Own Visualization")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            [
                "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", 
                "Box Plot", "Heatmap", "Pie Chart", "Area Chart"
            ]
        )
        
        column_types = st.session_state.column_types
        
        if viz_type == "Bar Chart":
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "X-Axis (Categories)",
                    [col for col, type_ in column_types.items() 
                     if type_ in ["categorical", "binary", "datetime"]]
                )
            
            with col2:
                y_col = st.selectbox(
                    "Y-Axis (Values)",
                    [col for col, type_ in column_types.items() 
                     if type_ in ["numeric"]]
                )
            
            color_col = st.selectbox(
                "Color By (Optional)",
                ["None"] + [col for col, type_ in column_types.items() 
                         if type_ in ["categorical", "binary"] and col != x_col]
            )
            
            # Generate plot
            if st.button("Generate Bar Chart"):
                if color_col != "None":
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"{y_col} by {x_col}"
                    )
                else:
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} by {x_col}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "X-Axis",
                    [col for col, type_ in column_types.items()]
                )
            
            with col2:
                y_col = st.selectbox(
                    "Y-Axis",
                    [col for col, type_ in column_types.items() 
                     if type_ in ["numeric"] and col != x_col]
                )
            
            color_col = st.selectbox(
                "Group By (Optional)",
                ["None"] + [col for col, type_ in column_types.items() 
                         if type_ in ["categorical", "binary"] and col != x_col and col != y_col]
            )
            
            # Generate plot
            if st.button("Generate Line Chart"):
                if color_col != "None":
                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"{y_col} vs {x_col}"
                    )
                else:
                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} vs {x_col}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "X-Axis",
                    [col for col, type_ in column_types.items() 
                     if type_ in ["numeric"]]
                )
            
            with col2:
                y_col = st.selectbox(
                    "Y-Axis",
                    [col for col, type_ in column_types.items() 
                     if type_ in ["numeric"] and col != x_col]
                )
            
            color_col = st.selectbox(
                "Color By (Optional)",
                ["None"] + [col for col, type_ in column_types.items() 
                         if col != x_col and col != y_col]
            )
            
            size_col = st.selectbox(
                "Size By (Optional)",
                ["None"] + [col for col, type_ in column_types.items() 
                         if type_ in ["numeric"] and col != x_col and col != y_col]
            )
            
            # Generate plot
            if st.button("Generate Scatter Plot"):
                if color_col != "None" and size_col != "None":
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        size=size_col,
                        title=f"{y_col} vs {x_col}"
                    )
                elif color_col != "None":
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"{y_col} vs {x_col}"
                    )
                elif size_col != "None":
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        size=size_col,
                        title=f"{y_col} vs {x_col}"
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} vs {x_col}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualization types can be implemented similarly
        
    else:
        st.info("Please upload a dataset in the Upload tab first.") 

elif st.session_state.current_tab == "Insights":
    st.header("💡 Data Insights")
    
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        column_types = st.session_state.column_types
        stats = st.session_state.stats
        correlations = st.session_state.correlations
        potential_targets = st.session_state.potential_targets
        insights = st.session_state.insights
        
        # Display insights
        st.subheader("Key Insights")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Correlations
        st.subheader("Top Correlations")
        
        if correlations:
            corr_data = []
            for (col1, col2), corr_value in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]:
                corr_data.append({
                    "Variable 1": col1,
                    "Variable 2": col2,
                    "Correlation": f"{corr_value:.2f}"
                })
            
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
            
            # Correlation heatmap
            numeric_cols = [col for col, type_ in column_types.items() if type_ in ["numeric", "binary"]]
            if len(numeric_cols) > 1:
                top_numeric = numeric_cols[:min(10, len(numeric_cols))]
                
                fig = plot_correlation_matrix(df, top_numeric)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant correlations found in the dataset.")
        
        # Potential targets
        if potential_targets:
            st.subheader("Potential Target Variables")
            st.write("These variables might be good prediction targets:")
            
            target_cols = st.columns(min(3, len(potential_targets)))
            for i, target in enumerate(potential_targets[:3]):
                with target_cols[i]:
                    st.info(target)
                    
                    # Show distribution for the target
                    if column_types[target] == "numeric":
                        fig = px.histogram(
                            df,
                            x=target,
                            title=f"Distribution of {target}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif column_types[target] in ["categorical", "binary"]:
                        counts = df[target].value_counts()
                        fig = px.pie(
                            values=counts.values,
                            names=counts.index,
                            title=f"Distribution of {target}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Column-specific insights
        st.subheader("Column Insights")
        
        selected_column = st.selectbox(
            "Select a column for detailed insights",
            list(df.columns)
        )
        
        if selected_column:
            col_type = column_types[selected_column]
            col_stats = stats[selected_column]
            
            st.write(f"### Analysis of {selected_column}")
            
            # Display basic stats
            st.markdown(f"**Type:** {col_type.capitalize()}")
            st.markdown(f"**Missing values:** {col_stats['missing_pct']:.1f}%")
            st.markdown(f"**Unique values:** {col_stats['unique_count']} ({col_stats['unique_pct']:.1f}%)")
            
            if col_type == "numeric":
                st.markdown(f"**Range:** {col_stats['min']:.2f} to {col_stats['max']:.2f}")
                st.markdown(f"**Mean:** {col_stats['mean']:.2f}")
                st.markdown(f"**Median:** {col_stats['median']:.2f}")
                st.markdown(f"**Standard deviation:** {col_stats['std']:.2f}")
                
                if col_stats.get('skew'):
                    skew = col_stats['skew']
                    if abs(skew) < 0.5:
                        skew_desc = "approximately symmetric"
                    elif abs(skew) < 1:
                        skew_desc = "moderately skewed"
                    else:
                        skew_desc = "highly skewed"
                    
                    st.markdown(f"**Skewness:** {skew:.2f} ({skew_desc})")
                
                # Show distribution
                fig = px.histogram(
                    df,
                    x=selected_column,
                    marginal="box",
                    title=f"Distribution of {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlations with this column
                related_cols = []
                for (col1, col2), corr_value in correlations.items():
                    if col1 == selected_column:
                        related_cols.append((col2, corr_value))
                    elif col2 == selected_column:
                        related_cols.append((col1, corr_value))
                
                if related_cols:
                    st.subheader(f"Columns correlated with {selected_column}")
                    
                    related_df = pd.DataFrame(
                        related_cols,
                        columns=["Column", "Correlation"]
                    ).sort_values("Correlation", ascending=False)
                    
                    st.dataframe(related_df, use_container_width=True)
                    
                    # Scatter plot with most correlated column
                    if related_cols:
                        most_correlated = related_cols[0][0]
                        fig = px.scatter(
                            df,
                            x=selected_column,
                            y=most_correlated,
                            title=f"{selected_column} vs {most_correlated} (Correlation: {related_cols[0][1]:.2f})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif col_type in ["categorical", "binary"]:
                # Show value counts
                value_counts = df[selected_column].value_counts()
                
                st.markdown(f"**Most common value:** {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
                
                if len(value_counts) > 1:
                    st.markdown(f"**Least common value:** {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)")
                
                # Show bar chart of distribution
                fig = px.bar(
                    x=value_counts.index[:20],  # Limit to top 20
                    y=value_counts.values[:20],
                    title=f"Distribution of {selected_column}",
                    labels={"x": selected_column, "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Relationship with numeric columns
                numeric_cols = [col for col, type_ in column_types.items() 
                              if type_ == "numeric" and col != selected_column]
                
                if numeric_cols:
                    st.subheader(f"{selected_column} vs Numeric Variables")
                    
                    selected_numeric = st.selectbox(
                        "Select a numeric column to compare",
                        numeric_cols
                    )
                    
                    if selected_numeric:
                        fig = px.box(
                            df,
                            x=selected_column,
                            y=selected_numeric,
                            title=f"{selected_numeric} by {selected_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif col_type == "datetime":
                # Show time-based analysis
                if not pd.api.types.is_datetime64_any_dtype(df[selected_column]):
                    date_series = pd.to_datetime(df[selected_column], errors='coerce')
                else:
                    date_series = df[selected_column]
                
                min_date = date_series.min()
                max_date = date_series.max()
                
                st.markdown(f"**Date range:** {min_date.date()} to {max_date.date()} ({(max_date - min_date).days} days)")
                
                # Distribution by month/year
                st.subheader("Time Distribution")
                
                # Group by month
                monthly_counts = date_series.dt.to_period('M').value_counts().sort_index()
                monthly_df = pd.DataFrame({
                    'Month': [str(date) for date in monthly_counts.index],
                    'Count': monthly_counts.values
                })
                
                fig = px.line(
                    monthly_df,
                    x='Month',
                    y='Count',
                    title=f"Distribution by month for {selected_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload a dataset in the Upload tab first.")

elif st.session_state.current_tab == "Custom":
    st.header("🎨 Custom Analysis")
    
    if st.session_state.dataframe is not None:
        df = st.session_state.dataframe
        column_types = st.session_state.column_types
        
        # Custom analysis options
        analysis_type = st.radio(
            "Choose Analysis Type",
            ["Custom Chart", "Data Filtering", "Statistical Test"],
            horizontal=True
        )
        
        if analysis_type == "Custom Chart":
            st.subheader("Create Custom Chart")
            
            chart_type = st.selectbox(
                "Select Chart Type",
                [
                    "Scatter", "Line", "Bar", "Histogram", "Box Plot",
                    "Violin Plot", "Heatmap", "Pie Chart", "3D Scatter"
                ]
            )
            
            # Get appropriate columns based on chart type
            if chart_type in ["Scatter", "Line"]:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox(
                        "X-Axis",
                        df.columns
                    )
                
                with col2:
                    y_col = st.selectbox(
                        "Y-Axis",
                        [col for col in df.columns if col != x_col]
                    )
                
                color_col = st.selectbox(
                    "Color By (Optional)",
                    ["None"] + [col for col in df.columns if col != x_col and col != y_col]
                )
                
                # Create plot
                if st.button("Generate Chart"):
                    if chart_type == "Scatter":
                        if color_col != "None":
                            fig = px.scatter(
                                df,
                                x=x_col,
                                y=y_col,
                                color=color_col,
                                title=f"{y_col} vs {x_col}"
                            )
                        else:
                            fig = px.scatter(
                                df,
                                x=x_col,
                                y=y_col,
                                title=f"{y_col} vs {x_col}"
                            )
                    else:  # Line chart
                        if color_col != "None":
                            fig = px.line(
                                df,
                                x=x_col,
                                y=y_col,
                                color=color_col,
                                title=f"{y_col} vs {x_col}"
                            )
                        else:
                            fig = px.line(
                                df,
                                x=x_col,
                                y=y_col,
                                title=f"{y_col} vs {x_col}"
                            )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Bar":
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox(
                        "X-Axis (Categories)",
                        [col for col, type_ in column_types.items() 
                         if type_ in ["categorical", "binary", "datetime"]]
                    )
                
                with col2:
                    y_col = st.selectbox(
                        "Y-Axis (Values)",
                        ["Count"] + [col for col, type_ in column_types.items() 
                                   if type_ == "numeric"]
                    )
                
                # Create plot
                if st.button("Generate Chart"):
                    if y_col == "Count":
                        # Simple count
                        value_counts = df[x_col].value_counts().reset_index()
                        value_counts.columns = [x_col, "Count"]
                        
                        fig = px.bar(
                            value_counts,
                            x=x_col,
                            y="Count",
                            title=f"Count of {x_col}"
                        )
                    else:
                        # Aggregated values
                        fig = px.bar(
                            df,
                            x=x_col,
                            y=y_col,
                            title=f"{y_col} by {x_col}"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Additional chart types can be implemented similarly
            
        elif analysis_type == "Data Filtering":
            st.subheader("Advanced Data Filtering")
            
            # Build filters
            st.write("Add filters to create a custom subset of your data")
            
            filters = []
            
            # Add filter button
            if st.button("Add Filter"):
                if 'filter_count' not in st.session_state:
                    st.session_state.filter_count = 1
                else:
                    st.session_state.filter_count += 1
            
            # Show existing filters
            for i in range(getattr(st.session_state, 'filter_count', 1)):
                with st.expander(f"Filter {i+1}"):
                    col_filter = st.selectbox(
                        "Column",
                        df.columns,
                        key=f"filter_{i}_col"
                    )
                    
                    col_type = column_types.get(col_filter, "unknown")
                    
                    if col_type == "numeric":
                        operation = st.selectbox(
                            "Operation",
                            ["equals", "greater than", "less than", "between"],
                            key=f"filter_{i}_op"
                        )
                        
                        if operation == "between":
                            min_val, max_val = st.slider(
                                "Value Range",
                                float(df[col_filter].min()),
                                float(df[col_filter].max()),
                                (float(df[col_filter].min()), float(df[col_filter].max())),
                                key=f"filter_{i}_val"
                            )
                            filters.append((col_filter, operation, (min_val, max_val)))
                        else:
                            value = st.number_input(
                                "Value",
                                value=float(df[col_filter].mean()),
                                key=f"filter_{i}_val"
                            )
                            filters.append((col_filter, operation, value))
                    
                    elif col_type in ["categorical", "binary", "text"]:
                        operation = st.selectbox(
                            "Operation",
                            ["equals", "not equals", "contains", "starts with", "in list"],
                            key=f"filter_{i}_op"
                        )
                        
                        if operation == "in list":
                            values = st.multiselect(
                                "Values",
                                df[col_filter].dropna().unique(),
                                key=f"filter_{i}_val"
                            )
                            filters.append((col_filter, operation, values))
                        else:
                            value = st.text_input(
                                "Value",
                                key=f"filter_{i}_val"
                            )
                            filters.append((col_filter, operation, value))
                    
                    elif col_type == "datetime":
                        operation = st.selectbox(
                            "Operation",
                            ["equals", "after", "before", "between"],
                            key=f"filter_{i}_op"
                        )
                        
                        if operation == "between":
                            start_date = st.date_input(
                                "Start Date",
                                key=f"filter_{i}_start"
                            )
                            end_date = st.date_input(
                                "End Date",
                                key=f"filter_{i}_end"
                            )
                            filters.append((col_filter, operation, (start_date, end_date)))
                        else:
                            date_val = st.date_input(
                                "Date",
                                key=f"filter_{i}_val"
                            )
                            filters.append((col_filter, operation, date_val))
            
            # Apply filters
            if st.button("Apply Filters"):
                filtered_df = df.copy()
                
                for col, operation, value in filters:
                    if column_types.get(col) == "numeric":
                        if operation == "equals":
                            filtered_df = filtered_df[filtered_df[col] == value]
                        elif operation == "greater than":
                            filtered_df = filtered_df[filtered_df[col] > value]
                        elif operation == "less than":
                            filtered_df = filtered_df[filtered_df[col] < value]
                        elif operation == "between":
                            min_val, max_val = value
                            filtered_df = filtered_df[(filtered_df[col] >= min_val) & 
                                                     (filtered_df[col] <= max_val)]
                    
                    elif column_types.get(col) in ["categorical", "binary", "text"]:
                        if operation == "equals":
                            filtered_df = filtered_df[filtered_df[col] == value]
                        elif operation == "not equals":
                            filtered_df = filtered_df[filtered_df[col] != value]
                        elif operation == "contains":
                            filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(value)]
                        elif operation == "starts with":
                            filtered_df = filtered_df[filtered_df[col].astype(str).str.startswith(value)]
                        elif operation == "in list":
                            filtered_df = filtered_df[filtered_df[col].isin(value)]
                    
                    elif column_types.get(col) == "datetime":
                        if not pd.api.types.is_datetime64_any_dtype(filtered_df[col]):
                            filtered_df[col] = pd.to_datetime(filtered_df[col], errors='coerce')
                        
                        if operation == "equals":
                            filtered_df = filtered_df[filtered_df[col].dt.date == value]
                        elif operation == "after":
                            filtered_df = filtered_df[filtered_df[col].dt.date > value]
                        elif operation == "before":
                            filtered_df = filtered_df[filtered_df[col].dt.date < value]
                        elif operation == "between":
                            start_date, end_date = value
                            filtered_df = filtered_df[(filtered_df[col].dt.date >= start_date) & 
                                                     (filtered_df[col].dt.date <= end_date)]
                
                st.write(f"Filtered data: {len(filtered_df)} rows (from original {len(df)} rows)")
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
        
        elif analysis_type == "Statistical Test":
            st.subheader("Statistical Tests")
            
            test_type = st.selectbox(
                "Select Test Type",
                ["Correlation Analysis", "Group Comparison", "Distribution Test"]
            )
            
            if test_type == "Correlation Analysis":
                numeric_cols = [col for col, type_ in column_types.items() if type_ == "numeric"]
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_col = st.selectbox(
                            "First Variable",
                            numeric_cols
                        )
                    
                    with col2:
                        y_col = st.selectbox(
                            "Second Variable",
                            [col for col in numeric_cols if col != x_col]
                        )
                    
                    corr_method = st.radio(
                        "Correlation Method",
                        ["Pearson", "Spearman"],
                        horizontal=True
                    )
                    
                    if st.button("Calculate Correlation"):
                        if corr_method == "Pearson":
                            corr, p_value = stats.pearsonr(df[x_col].dropna(), df[y_col].dropna())
                        else:
                            corr, p_value = stats.spearmanr(df[x_col].dropna(), df[y_col].dropna())
                        
                        st.metric("Correlation Coefficient", f"{corr:.4f}")
                        st.metric("P-value", f"{p_value:.4f}")
                        
                        # Interpret result
                        if p_value < 0.05:
                            st.success(f"Statistically significant correlation detected (p < 0.05)")
                        else:
                            st.info(f"No statistically significant correlation (p > 0.05)")
                        
                        # Scatter plot
                        fig = px.scatter(
                            df,
                            x=x_col,
                            y=y_col,
                            trendline="ols",
                            title=f"{y_col} vs {x_col} (Correlation: {corr:.4f})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least two numeric columns for correlation analysis")
            
            # Additional statistical tests can be implemented similarly
                
    else:
        st.info("Please upload a dataset in the Upload tab first.")

# Run the app
if __name__ == "__main__":
    # This will be executed when the script is run directly
    pass 