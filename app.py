from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import os
import io
import uuid
from werkzeug.utils import secure_filename
import re
from collections import Counter
import time
from sklearn.preprocessing import LabelEncoder
import base64
from datetime import datetime
from together import Together
import asyncio
import threading
import tabula
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
import pandas as pd
import tempfile

app = Flask(__name__)
app.secret_key = 'data_viz_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize Together API client
try:
    together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY', ''))
except Exception as e:
    print(f"Error initializing Together API: {str(e)}")
    together_client = None

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    
    # 9. Target variable analysis if potential targets exist
    if potential_targets and (numeric_cols or cat_cols):
        target = potential_targets[0]
        predictors = [col for col in numeric_cols + cat_cols if col != target][:5]
        if predictors:
            recommendations.append({
                "type": "target_analysis",
                "title": f"Predictive Analysis for {target}",
                "target": target,
                "predictors": predictors,
                "description": "Explore variables that might predict the target"
            })
    
    # 10. 3D scatter plot if at least 3 numeric columns
    if len(numeric_cols) >= 3:
        recommendations.append({
            "type": "scatter3d",
            "title": "3D Relationship Visualization",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "z": numeric_cols[2],
            "description": "Explore complex relationships in three dimensions"
        })
    
    return recommendations[:10]  # Return top 10 recommendations

# LLM-powered functions
def generate_llm_insights(df, column_types, stats, correlations, potential_targets):
    """Generate insights using LLM"""
    if not together_client:
        return ["API key required for LLM insights. Please set the TOGETHER_API_KEY environment variable."]
    
    try:
        # Prepare dataset summary for the LLM
        data_summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_types": column_types,
            "column_stats": {k: v for k, v in stats.items() if k in list(df.columns)[:10]}  # Limit to first 10 columns
        }
        
        # Sample data (first 5 rows, first 10 columns)
        sample_data = df.iloc[:5, :10].to_dict(orient="records")
        
        # Correlations
        top_correlations = []
        for (col1, col2), corr_value in sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]:
            top_correlations.append(f"{col1} and {col2}: {corr_value:.2f}")
        
        prompt = f"""
        As a data scientist, analyze this dataset and provide key insights:
        
        Dataset summary: 
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        - Column types: {json.dumps({k: v for k, v in column_types.items() if k in list(df.columns)[:10]})}
        
        Top correlations:
        {chr(10).join(top_correlations) if top_correlations else "No significant correlations found"}
        
        Sample data: 
        {json.dumps(sample_data, default=str)}
        
        Provide 5-7 key observations about this data and suggest potential analysis directions.
        Focus on patterns, relationships, and potential business implications.
        Be specific and insightful rather than generic.
        Format your response as a list of bullet points.
        """
        
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a data science expert who provides concise, insightful analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.5,
            top_p=0.7,
            top_k=50
        )
        
        insights_text = response.choices[0].message.content
        
        # Convert to list of insights
        insights = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                insights.append(line[2:])
            elif line.startswith('* '):
                insights.append(line[2:])
            elif line and not line.startswith('#') and len(line) > 20:  # Skip headers, keep substantial lines
                insights.append(line)
        
        # Filter out empty lines and ensure we have insights
        insights = [insight for insight in insights if insight]
        
        if not insights:
            insights = ["The LLM couldn't generate specific insights. This might be due to limited data or complexity."]
        
        return insights
    except Exception as e:
        print(f"Error generating LLM insights: {str(e)}")
        return [f"Error generating insights: {str(e)}"]

def generate_llm_chart_ideas(df, column_types):
    """Generate chart ideas using LLM"""
    if not together_client:
        return []
    
    try:
        # Prepare column information
        numeric_cols = [col for col, type_ in column_types.items() if type_ in ["numeric", "binary"]]
        cat_cols = [col for col, type_ in column_types.items() if type_ in ["categorical"]]
        datetime_cols = [col for col, type_ in column_types.items() if type_ in ["datetime"]]
        
        prompt = f"""
        As a data visualization expert, suggest creative and insightful chart ideas for this dataset:
        
        Dataset information:
        - Numeric columns: {', '.join(numeric_cols[:10]) if numeric_cols else 'None'}
        - Categorical columns: {', '.join(cat_cols[:10]) if cat_cols else 'None'}
        - Datetime columns: {', '.join(datetime_cols) if datetime_cols else 'None'}
        
        For each chart idea:
        1. Provide a descriptive title
        2. Specify the chart type (bar, line, scatter, etc.)
        3. List the columns to use
        4. Explain what insights this visualization might reveal
        
        Suggest 5 different visualization ideas that would provide unique insights.
        Format each idea as a JSON object with the following structure:
        {{
            "title": "Chart title",
            "type": "chart_type",
            "columns": ["column1", "column2"],
            "description": "What insights this visualization might reveal"
        }}
        
        Return a JSON array of these objects.
        """
        
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a data visualization expert who provides creative and insightful chart ideas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        chart_ideas_text = response.choices[0].message.content
        
        # Extract JSON array from response
        try:
            # Find JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', chart_ideas_text, re.DOTALL)
            if json_match:
                chart_ideas = json.loads(json_match.group(0))
            else:
                # Try to find individual JSON objects and combine them
                json_objects = re.findall(r'\{\s*"title".*?\}', chart_ideas_text, re.DOTALL)
                if json_objects:
                    chart_ideas = [json.loads(obj) for obj in json_objects]
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Validate chart ideas
            valid_chart_ideas = []
            for idea in chart_ideas:
                if isinstance(idea, dict) and "title" in idea and "type" in idea and "columns" in idea and "description" in idea:
                    # Ensure columns exist in the dataset
                    idea["columns"] = [col for col in idea["columns"] if col in df.columns]
                    if idea["columns"]:  # Only add if there are valid columns
                        valid_chart_ideas.append(idea)
            
            return valid_chart_ideas
        except Exception as e:
            print(f"Error parsing chart ideas JSON: {str(e)}")
            print(f"Raw response: {chart_ideas_text}")
            return []
    except Exception as e:
        print(f"Error generating chart ideas: {str(e)}")
        return []

def get_llm_response_for_chat(question, df_info):
    """Get LLM response for chat questions"""
    if not together_client:
        return "API key required for chat. Please set the TOGETHER_API_KEY environment variable."
    
    try:
        prompt = f"""
        Dataset information:
        {df_info}
        
        User question: {question}
        
        Provide a helpful, concise answer based on the dataset information.
        Focus on giving specific insights related to the data.
        If you cannot answer based on the provided information, explain what additional data would be needed.
        """
        
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant who provides concise, accurate answers about datasets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
            top_p=0.9,
            top_k=50
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting chat response: {str(e)}")
        return f"Error: {str(e)}"

def generate_llm_column_insights(df, column_name, column_type, stats):
    """Generate insights for a specific column using LLM"""
    if not together_client:
        return "API key required for column insights."
    
    try:
        # Prepare column information
        column_info = {
            "name": column_name,
            "type": column_type,
            "stats": stats[column_name]
        }
        
        # Get sample values
        if column_type in ["categorical", "binary"]:
            sample_values = df[column_name].value_counts().head(10).to_dict()
            column_info["sample_values"] = {str(k): int(v) for k, v in sample_values.items()}
        elif column_type == "numeric":
            column_info["histogram"] = np.histogram(df[column_name].dropna(), bins=10)[0].tolist()
        
        prompt = f"""
        Analyze this column from a dataset and provide insights:
        
        Column name: {column_name}
        Column type: {column_type}
        Statistics: {json.dumps(stats[column_name], default=str)}
        
        {json.dumps({"sample_values": column_info.get("sample_values", {})}, default=str) if "sample_values" in column_info else ""}
        {json.dumps({"histogram": column_info.get("histogram", [])}, default=str) if "histogram" in column_info else ""}
        
        Provide 3-5 specific insights about this column, such as:
        - Distribution patterns
        - Potential issues (outliers, skewness, etc.)
        - Business implications
        - Relationships to explore with other columns
        
        Format your response as a list of bullet points.
        """
        
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a data analysis expert who provides concise, insightful analysis of dataset columns."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.5,
            top_p=0.7,
            top_k=50
        )
        
        insights_text = response.choices[0].message.content
        
        # Convert to list of insights
        insights = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                insights.append(line[2:])
            elif line.startswith('* '):
                insights.append(line[2:])
            elif line and not line.startswith('#') and len(line) > 20:  # Skip headers, keep substantial lines
                insights.append(line)
        
        # Filter out empty lines and ensure we have insights
        insights = [insight for insight in insights if insight]
        
        if not insights:
            insights = ["No specific insights could be generated for this column."]
        
        return insights
    except Exception as e:
        print(f"Error generating column insights: {str(e)}")
        return [f"Error generating insights: {str(e)}"]

# Visualization functions
def create_visualization(df, viz_type, config):
    """Create visualization based on type and configuration"""
    try:
        if viz_type == "distribution":
            return plot_distributions(df, config["columns"])
        elif viz_type == "categorical_counts":
            return plot_categorical_counts(df, config["columns"])
        elif viz_type == "correlation":
            return plot_correlation_matrix(df, config["columns"])
        elif viz_type == "scatter":
            return plot_scatter(df, config["x"], config["y"])
        elif viz_type == "boxplot":
            return plot_boxplot(df, config["cat_column"], config["num_column"])
        elif viz_type == "missing_values":
            return plot_missing_values(df, config["columns"])
        elif viz_type == "time_series":
            return plot_time_series(df, config["time_column"], config["value_column"])
        elif viz_type == "pairplot":
            return plot_pairplot(df, config["columns"])
        elif viz_type == "target_analysis":
            return plot_target_analysis(df, config["target"], config["predictors"])
        elif viz_type == "scatter3d":
            return plot_scatter3d(df, config["x"], config["y"], config["z"])
        else:
            return None
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

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
                histnorm='probability density',
                marker_color=px.colors.qualitative.Plotly[columns.index(col) % len(px.colors.qualitative.Plotly)]
            ))
        except Exception as e:
            print(f"Couldn't plot distribution for {col}: {str(e)}")
            
    fig.update_layout(
        title="Distribution of Numeric Features",
        xaxis_title="Value",
        yaxis_title="Density",
        barmode='overlay',
        height=500,
        template="plotly_white"
    )
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_categorical_counts(df, columns):
    """Plot counts of categorical variables"""
    if len(columns) == 1:
        # Single column
        col = columns[0]
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        
        # Limit to top 15 categories if there are many
        if len(value_counts) > 15:
            value_counts = value_counts.head(15)
            title = f"Top 15 Categories in {col}"
        else:
            title = f"Count of {col}"
        
        fig = px.bar(
            value_counts, 
            x=col, 
            y='count',
            title=title,
            color=col,
            template="plotly_white"
        )
    else:
        # Multiple columns - create subplots
        fig = go.Figure()
        for i, col in enumerate(columns[:4]):  # Limit to 4 categories
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, 'count']
            
            # Limit to top 10 categories if there are many
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
            
            fig.add_trace(go.Bar(
                x=value_counts[col],
                y=value_counts['count'],
                name=col,
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            ))
            
        fig.update_layout(
            title="Categorical Feature Counts",
            xaxis_title="Category",
            yaxis_title="Count",
            barmode='group',
            height=500,
            template="plotly_white"
        )
    
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

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
        title="Correlation Matrix",
        template="plotly_white"
    )
    
    fig.update_layout(height=600)
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_scatter(df, x_col, y_col):
    """Create scatter plot between two variables"""
    # Check if we should color by a categorical column
    cat_cols = [col for col in df.columns if df[col].nunique() <= 10 and col != x_col and col != y_col]
    color_col = cat_cols[0] if cat_cols else None
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        color=color_col,
        title=f"Relationship: {x_col} vs {y_col}" + (f" (colored by {color_col})" if color_col else ""),
        trendline="ols",  # Add trend line
        opacity=0.7,
        template="plotly_white"
    )
    
    # Add correlation annotation
    corr_val = df[[x_col, y_col]].corr().iloc[0, 1]
    fig.add_annotation(
        x=0.95,
        y=0.05,
        text=f"Correlation: {corr_val:.2f}",
        showarrow=False,
        xref="paper",
        yref="paper",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.5)",
        borderwidth=1,
        borderpad=4
    )
    
    fig.update_layout(height=500)
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_boxplot(df, cat_col, num_col):
    """Create box plot for numeric column across categories"""
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
        color=cat_col,
        template="plotly_white",
        points="outliers"  # Only show outlier points
    )
    
    fig.update_layout(
        height=500,
        xaxis={'categoryorder':'total descending'}  # Order by frequency
    )
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_missing_values(df, columns):
    """Visualize missing values patterns"""
    # Calculate missing percentages
    missing_pct = df[columns].isna().mean().round(3) * 100
    missing_df = pd.DataFrame({
        'Column': missing_pct.index,
        'Missing %': missing_pct.values
    }).sort_values('Missing %', ascending=False)
    
    fig = px.bar(
        missing_df,
        x='Column',
        y='Missing %',
        title="Missing Values by Column",
        template="plotly_white",
        color='Missing %',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=500)
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_time_series(df, time_col, value_col):
    """Create time series plot"""
    # Ensure time column is datetime
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Group by time and calculate mean if needed
        if df[time_col].nunique() > 100:
            # Resample to appropriate frequency
            df_grouped = df.groupby(pd.Grouper(key=time_col, freq='D')).mean().reset_index()
            resampled_note = " (Daily Resampled)"
        else:
            df_grouped = df.copy()
            resampled_note = ""
        
        # Create line chart
        fig = px.line(
            df_grouped.sort_values(time_col),
            x=time_col,
            y=value_col,
            title=f"{value_col} over Time{resampled_note}",
            markers=True,
            template="plotly_white"
        )
        
        fig.update_layout(height=500)
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    except Exception as e:
        print(f"Error in time series plot: {str(e)}")
        # Fallback to scatter plot
        fig = px.scatter(
            df, 
            x=time_col, 
            y=value_col, 
            title=f"{value_col} by {time_col}",
            template="plotly_white"
        )
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_pairplot(df, columns):
    """Create pairplot matrix using Plotly"""
    # Check if we should color by a categorical column
    cat_cols = [col for col in df.columns if col not in columns and df[col].nunique() <= 10]
    color_col = cat_cols[0] if cat_cols else None
    
    # Create a grid of scatter plots
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color_col,
        title="Multi-variable Relationships" + (f" (colored by {color_col})" if color_col else ""),
        opacity=0.7,
        template="plotly_white"
    )
    
    fig.update_layout(height=700)
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_target_analysis(df, target, predictors):
    """Create visualizations to analyze relationship with target variable"""
    # Determine if target is categorical or numerical
    is_target_cat = df[target].dtype == 'object' or df[target].nunique() < 10
    
    if is_target_cat:
        # For categorical target, show distribution against each predictor
        # If there's a numeric predictor, use box plot
        numeric_predictors = [p for p in predictors if pd.api.types.is_numeric_dtype(df[p]) and df[p].nunique() > 10]
        if numeric_predictors:
            predictor = numeric_predictors[0]
            
            # Limit to top 8 categories if there are many
            if df[target].nunique() > 8:
                top_cats = df[target].value_counts().nlargest(8).index.tolist()
                df_plot = df[df[target].isin(top_cats)].copy()
                title = f"Top 8 Categories: Distribution of {predictor} by {target}"
            else:
                df_plot = df.copy()
                title = f"Distribution of {predictor} by {target}"
                
            fig = px.box(
                df_plot, 
                x=target, 
                y=predictor,
                color=target,
                title=title,
                template="plotly_white",
                points="outliers"
            )
            return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        else:
            # If no numeric predictors, use categorical comparison
            cat_predictor = predictors[0]
            
            # Limit to top categories if there are many
            if df[target].nunique() > 8 or df[cat_predictor].nunique() > 8:
                top_target_cats = df[target].value_counts().nlargest(8).index.tolist()
                top_pred_cats = df[cat_predictor].value_counts().nlargest(8).index.tolist()
                df_plot = df[df[target].isin(top_target_cats) & df[cat_predictor].isin(top_pred_cats)].copy()
                title = f"Top Categories: Relationship between {target} and {cat_predictor}"
            else:
                df_plot = df.copy()
                title = f"Relationship between {target} and {cat_predictor}"
                
            crosstab = pd.crosstab(df_plot[target], df_plot[cat_predictor])
            
            # Normalize to show percentages
            crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            
            # Create heatmap
            fig = px.imshow(
                crosstab_pct,
                text_auto='.1f',
                labels=dict(x=cat_predictor, y=target, color="Percentage (%)"),
                title=title,
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            
            fig.update_layout(height=500)
            return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    else:
        # For numerical target, use scatter plots with regression line
        predictor = predictors[0]
        fig = px.scatter(
            df, 
            x=predictor, 
            y=target,
            trendline="ols",
            title=f"Relationship between {predictor} and {target}",
            template="plotly_white",
            opacity=0.7
        )
        
        # Add correlation value
        corr_val = df[[target, predictor]].corr().iloc[0, 1]
        fig.add_annotation(
            x=0.95,
            y=0.95,
            text=f"Correlation: {corr_val:.2f}",
            showarrow=False,
            xref="paper",
            yref="paper",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
            borderpad=4
        )
    
        fig.update_layout(height=500)
        return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def plot_scatter3d(df, x_col, y_col, z_col):
    """Create 3D scatter plot"""
    # Check if we should color by a categorical column
    cat_cols = [col for col in df.columns 
                if col not in [x_col, y_col, z_col] and df[col].nunique() <= 10]
    color_col = cat_cols[0] if cat_cols else None
    
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        title=f"3D Relationship: {x_col}, {y_col}, {z_col}" + (f" (colored by {color_col})" if color_col else ""),
        opacity=0.7,
        template="plotly_white"
    )
    
    fig.update_layout(height=700)
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

# Modify the upload_file function to include more preprocessing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Check file extension
            if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
                return jsonify({'error': 'File type not supported. Please upload a CSV or Excel file.'}), 400
            
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Generate a unique session ID for this dataset
            session_id = str(uuid.uuid4())
            
            # Load and process the data
            try:
                if filename.endswith('.csv'):
                    # Try to detect proper data types by first reading with string types
                    df = pd.read_csv(file_path, dtype=str)
                else:
                    # For Excel files, also read everything as strings first
                    df = pd.read_excel(file_path, dtype=str)
                    
                # Preprocess the data
                # 1. Handle missing values - replace with appropriate values or drop if too many
                for col in df.columns:
                    missing_pct = df[col].isna().mean() * 100
                    if missing_pct > 0 and missing_pct < 30:  # If missing values are less than 30%
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # Fill numeric columns with median
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            # Fill categorical columns with mode
                            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                            df[col] = df[col].fillna(mode_value)
                
                # 2. Proper type detection and conversion
                for col in df.columns:
                    # First check column name for date indicators
                    date_keywords = ['date', 'time', 'day', 'month', 'year']
                    is_likely_date_by_name = any(keyword in col.lower() for keyword in date_keywords)
                    
                    # Check if column has date-like format but only if it consistently looks like dates
                    sample = df[col].dropna().head(10).tolist()
                    date_count = 0
                    
                    # Common date patterns
                    date_patterns = [
                        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',  # YYYY-MM-DD or YYYY/MM/DD
                        r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',   # DD-MM-YYYY or MM-DD-YYYY
                        r'^\d{1,2}[-/]\w{3}[-/]\d{4}$'      # DD-MMM-YYYY
                    ]
                    
                    for value in sample:
                        if any(re.match(pattern, str(value)) for pattern in date_patterns):
                            date_count += 1
                    
                    # Only convert to datetime if:
                    # 1. The column name indicates it's a date column, OR
                    # 2. Most values match date patterns exactly
                    if is_likely_date_by_name or (date_count > len(sample) * 0.8):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            continue
                        except:
                            pass  # If conversion fails, try numeric or categorical
                    
                    # Check if column can be converted to numeric
                    try:
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        if numeric_values.notna().sum() > df[col].count() * 0.7:  # If > 70% convert successfully
                            df[col] = numeric_values
                            continue
                    except:
                        pass
                    
                    # If not date or numeric, treat as categorical
                    if df[col].nunique() < min(50, df.shape[0] * 0.2):  # If relatively few unique values
                        df[col] = df[col].astype('category')
                
                # 3. Handle outliers in numeric columns
                for col in df.select_dtypes(include=['number']).columns:
                    # Calculate IQR
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower_bound, upper_bound)
                
                # Save the preprocessed data
                preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"preprocessed_{filename}")
                if preprocessed_path.endswith('.csv'):
                    df.to_csv(preprocessed_path, index=False)
                else:
                    df.to_excel(preprocessed_path, index=False)
                
                # Store the preprocessed file path in the session
                session['file_path'] = preprocessed_path
                session['original_file_path'] = file_path
                session['session_id'] = session_id
            except Exception as e:
                return jsonify({'error': f'Error reading or preprocessing file: {str(e)}'}), 500
            
            # Store basic info about the dataset
            session['rows'] = len(df)
            session['columns'] = len(df.columns)
            session['column_names'] = df.columns.tolist()
            
            # Analyze the data
            try:
                column_types = analyze_column_types(df)
                stats = get_column_stats(df)
                correlations = identify_correlations(df, column_types)
                potential_targets = identify_potential_targets(df, column_types)
                
                # Generate visualization recommendations
                recommendations = generate_visualization_recommendations(df, column_types, stats, correlations, potential_targets)
                
                # Generate LLM insights
                llm_insights = generate_llm_insights(df, column_types, stats, correlations, potential_targets)
                
                # Generate LLM chart ideas
                llm_chart_ideas = generate_llm_chart_ideas(df, column_types)
                
                # Store analysis results in session
                session['column_types'] = column_types
                session['stats'] = stats
                session['correlations'] = correlations
                session['potential_targets'] = potential_targets
                session['recommendations'] = recommendations
                session['llm_insights'] = llm_insights
                session['llm_chart_ideas'] = llm_chart_ideas
                
                # Create dataset info for chat
                df_info = f"""
                Dataset: {filename}
                Rows: {len(df)}
                Columns: {len(df.columns)}
                Column types: {json.dumps(column_types)}
                """
                session['df_info'] = df_info
                
                # Add preprocessing info
                preprocessing_info = {
                    "missing_values_handled": True,
                    "date_columns_converted": True,
                    "outliers_handled": True
                }
                session['preprocessing_info'] = preprocessing_info
                
            except Exception as e:
                return jsonify({'error': f'Error analyzing data: {str(e)}'}), 500
            
            return jsonify({
                'success': True,
                'redirect': url_for('overview')
            })
        except Exception as e:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File upload failed. Please try again.'}), 400

@app.route('/overview')
def overview():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    file_path = session.get('file_path')
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Get basic stats
        rows = session.get('rows')
        columns = session.get('columns')
        column_names = session.get('column_names')
        column_types = session.get('column_types')
        stats = session.get('stats')
        
        # Calculate missing values percentage
        missing_pct = sum(stats[col]['missing_count'] for col in stats) / (rows * columns) * 100
        
        # Count column types
        col_type_counts = Counter(column_types.values())
        
        # Get columns with highest missing values
        missing_cols = {col: stats[col]["missing_pct"] for col in column_names if stats[col]["missing_pct"] > 0}
        top_missing = sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get insights
        correlations = session.get('correlations')
        potential_targets = session.get('potential_targets')
        insights = session.get('llm_insights', generate_insights(df, column_types, stats, correlations, potential_targets))
        
        # Sample data
        sample_data = df.head(10).to_html(classes='table table-striped table-hover', index=False)
        
        # LLM chart ideas
        llm_chart_ideas = session.get('llm_chart_ideas', [])
        
        return render_template('overview.html', 
                              rows=rows,
                              columns=columns,
                              missing_pct=missing_pct,
                              col_type_counts=col_type_counts,
                              top_missing=top_missing,
                              insights=insights,
                              sample_data=sample_data,
                              llm_chart_ideas=llm_chart_ideas)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/explorer')
def explorer():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    file_path = session.get('file_path')
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        column_types = session.get('column_types', {})
        stats = session.get('stats', {})
        
        # Group columns by type
        columns_by_type = {}
        for col, type_ in column_types.items():
            if type_ not in columns_by_type:
                columns_by_type[type_] = []
            columns_by_type[type_].append(col)
        
        return render_template('explorer.html', 
                              columns_by_type=columns_by_type,
                              stats=stats,
                              column_types=column_types,
                              column_names=df.columns.tolist())
    except Exception as e:
        print(f"Error in explorer route: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/visualizations')
def visualizations():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    recommendations = session.get('recommendations')
    llm_chart_ideas = session.get('llm_chart_ideas', [])
    
    return render_template('visualizations.html', 
                          recommendations=recommendations,
                          llm_chart_ideas=llm_chart_ideas)

@app.route('/insights')
def insights():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    file_path = session.get('file_path')
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        column_types = session.get('column_types')
        stats = session.get('stats')
        correlations = session.get('correlations')
        potential_targets = session.get('potential_targets')
        
        # Get LLM insights
        llm_insights = session.get('llm_insights', [])
        
        # If no LLM insights, generate basic insights
        if not llm_insights:
            llm_insights = generate_insights(df, column_types, stats, correlations, potential_targets)
        
        # Top correlations
        top_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Dataset info for chat
        df_info = session.get('df_info', '')
        
        return render_template('insights.html', 
                              insights=llm_insights,
                              top_correlations=top_correlations,
                              potential_targets=potential_targets,
                              df_info=df_info)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/custom')
def custom():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    file_path = session.get('file_path')
    column_types = session.get('column_types')
    llm_chart_ideas = session.get('llm_chart_ideas', [])
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Group columns by type
        columns_by_type = {}
        for col, type_ in column_types.items():
            if type_ not in columns_by_type:
                columns_by_type[type_] = []
            columns_by_type[type_].append(col)
        
        return render_template('custom.html', 
                              columns_by_type=columns_by_type,
                              llm_chart_ideas=llm_chart_ideas)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/get_column_data', methods=['POST'])
def get_column_data():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    file_path = session.get('file_path')
    column = request.json.get('column')
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        if column not in df.columns:
            return jsonify({'error': 'Column not found'}), 400
        
        # Get column data
        if pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns, return histogram data
            hist, bin_edges = np.histogram(df[column].dropna(), bins=20)
            return jsonify({
                'type': 'numeric',
                'data': {
                    'values': hist.tolist(),
                    'bins': bin_edges.tolist()
                },
                'stats': {
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'mean': float(df[column].mean()),
                    'median': float(df[column].median()),
                    'std': float(df[column].std())
                }
            })
        else:
            # For categorical columns, return value counts
            value_counts = df[column].value_counts().head(20)
            return jsonify({
                'type': 'categorical',
                'data': {
                    'labels': value_counts.index.tolist(),
                    'values': value_counts.values.tolist()
                },
                'stats': {
                    'unique': int(df[column].nunique()),
                    'top': str(df[column].value_counts().index[0]),
                    'freq': int(df[column].value_counts().iloc[0])
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_column_insights', methods=['POST'])
def get_column_insights():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    file_path = session.get('file_path')
    column = request.json.get('column')
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        if column not in df.columns:
            return jsonify({'error': 'Column not found'}), 400
        
        column_types = session.get('column_types')
        stats = session.get('stats')
        
        # Generate insights for the column
        insights = generate_llm_column_insights(df, column, column_types[column], stats)
        
        return jsonify({
            'success': True,
            'insights': insights
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_visualization', methods=['POST'])
def get_visualization():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    file_path = session.get('file_path')
    viz_type = request.json.get('type')
    config = request.json.get('config')
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Create visualization
        viz_data = create_visualization(df, viz_type, config)
        
        if viz_data:
            return jsonify({'success': True, 'data': viz_data})
        else:
            return jsonify({'error': 'Failed to create visualization'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter_data', methods=['POST'])
def filter_data():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    file_path = session.get('file_path')
    filters = request.json.get('filters', {})
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Apply filters
        filtered_df = df.copy()
        for col, filter_val in filters.items():
            if col not in filtered_df.columns:
                continue
                
            if pd.api.types.is_numeric_dtype(filtered_df[col]):
                if 'min' in filter_val and 'max' in filter_val:
                    filtered_df = filtered_df[(filtered_df[col] >= filter_val['min']) & 
                                             (filtered_df[col] <= filter_val['max'])]
            elif 'values' in filter_val and filter_val['values']:
                filtered_df = filtered_df[filtered_df[col].isin(filter_val['values'])]
        
        # Return basic info about filtered data
        return jsonify({
            'success': True,
            'rows': len(filtered_df),
            'sample': filtered_df.head(10).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_custom_viz', methods=['POST'])
def create_custom_viz():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    file_path = session.get('file_path')
    chart_type = request.json.get('chart_type')
    config = request.json.get('config', {})
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Create visualization based on chart type
        fig = None
        
        if chart_type == 'bar':
            x_col = config.get('x')
            y_col = config.get('y')
            color_col = config.get('color')
            
            if not x_col or not y_col:
                return jsonify({'error': 'Missing required columns'}), 400
                
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col, 
                color=color_col if color_col != 'None' else None,
                title=f"{y_col} by {x_col}",
                template="plotly_white"
            )
            
        elif chart_type == 'line':
            x_col = config.get('x')
            y_cols = config.get('y', [])
            
            if not x_col or not y_cols:
                return jsonify({'error': 'Missing required columns'}), 400
                
            fig = go.Figure()
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=df[x_col], 
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col
                ))
            
            fig.update_layout(
                title=f"Line Chart: {', '.join(y_cols)} by {x_col}",
                xaxis_title=x_col,
                yaxis_title="Value",
                template="plotly_white"
            )
                
        elif chart_type == 'scatter':
            x_col = config.get('x')
            y_col = config.get('y')
            color_col = config.get('color')
            
            if not x_col or not y_col:
                return jsonify({'error': 'Missing required columns'}), 400
                
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col, 
                color=color_col if color_col != 'None' else None,
                title=f"Scatter Plot: {y_col} vs {x_col}",
                template="plotly_white"
            )
            
        elif chart_type == 'box':
            x_col = config.get('x')
            y_col = config.get('y')
            
            if not x_col or not y_col:
                return jsonify({'error': 'Missing required columns'}), 400
                
            fig = px.box(
                df, 
                x=x_col, 
                y=y_col,
                color=x_col,
                title=f"Box Plot: {y_col} by {x_col}",
                template="plotly_white"
            )
            
        elif chart_type == 'histogram':
            col = config.get('column')
            bins = config.get('bins', 30)
            
            if not col:
                return jsonify({'error': 'Missing required column'}), 400
                
            fig = px.histogram(
                df, 
                x=col,
                nbins=bins,
                title=f"Histogram of {col}",
                template="plotly_white"
            )
            
        elif chart_type == 'pie':
            col = config.get('column')
            
            if not col:
                return jsonify({'error': 'Missing required column'}), 400
                
            # Limit to top categories if there are many
            if df[col].nunique() > 15:
                top_n = config.get('top_n', 10)
                top_cats = df[col].value_counts().nlargest(top_n).index.tolist()
                df_pie = df[df[col].isin(top_cats)].copy()
                title = f"Pie Chart: Top {top_n} Categories of {col}"
            else:
                df_pie = df.copy()
                title = f"Pie Chart: {col}"
                
            fig = px.pie(
                df_pie, 
                names=col,
                title=title,
                template="plotly_white"
            )
            
        elif chart_type == 'heatmap':
            cols = config.get('columns', [])
            
            if len(cols) < 2:
                return jsonify({'error': 'Need at least 2 columns for heatmap'}), 400
                
            corr_df = df[cols].corr()
            fig = px.imshow(
                corr_df,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                template="plotly_white"
            )
        
        if fig:
            return jsonify({
                'success': True, 
                'data': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
            })
        else:
            return jsonify({'error': 'Failed to create visualization'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_filtered', methods=['POST'])
def download_filtered():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    file_path = session.get('file_path')
    filters = request.json.get('filters', {})
    
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Apply filters
        filtered_df = df.copy()
        for col, filter_val in filters.items():
            if col not in filtered_df.columns:
                continue
                
            if pd.api.types.is_numeric_dtype(filtered_df[col]):
                if 'min' in filter_val and 'max' in filter_val:
                    filtered_df = filtered_df[(filtered_df[col] >= filter_val['min']) & 
                                             (filtered_df[col] <= filter_val['max'])]
            elif 'values' in filter_val and filter_val['values']:
                filtered_df = filtered_df[filtered_df[col].isin(filter_val['values'])]
        
        # Create CSV in memory
        output = io.BytesIO()
        filtered_df.to_csv(output, index=False)
        output.seek(0)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"filtered_data_{timestamp}.csv"
        
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    question = request.json.get('question', '')
    df_info = session.get('df_info', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Get response from LLM
        response = get_llm_response_for_chat(question, df_info)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a fix for the visualization chart generation issue by fixing the create_llm_viz function

# Fix the create_llm_viz function to properly handle chart generation
# Replace the entire function with this improved version

def create_llm_viz(df, idea_index):
    """Create visualization based on LLM chart idea"""
    try:
        # Get chart ideas
        llm_chart_ideas = session.get('llm_chart_ideas', [])
        
        if not llm_chart_ideas or idea_index >= len(llm_chart_ideas):
            return None, "Chart idea not found"
        
        # Get the selected chart idea
        chart_idea = llm_chart_ideas[idea_index]
        
        # Map chart type to visualization function
        chart_type_mapping = {
            'bar': 'bar',
            'line': 'line',
            'scatter': 'scatter',
            'pie': 'pie',
            'box': 'box',
            'histogram': 'histogram',
            'heatmap': 'heatmap',
            'correlation': 'correlation',
            'distribution': 'distribution',
            'boxplot': 'boxplot'
        }
        
        # Determine chart type and config
        chart_type = chart_type_mapping.get(chart_idea['type'].lower(), 'bar')
        
        # Create config based on chart type
        config = {}
        columns = chart_idea.get('columns', [])
        # Ensure columns exist in dataframe
        valid_columns = [col for col in columns if col in df.columns]
        
        if not valid_columns:
            return None, "No valid columns found for this chart"
        
        if chart_type == 'bar':
            if len(valid_columns) >= 2:
                config = {
                    'x': valid_columns[0],
                    'y': valid_columns[1],
                    'color': valid_columns[2] if len(valid_columns) > 2 else None
                }
                
                fig = px.bar(
                    df,
                    x=config['x'],
                    y=config['y'],
                    color=config['color'],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
            elif len(valid_columns) == 1:
                # Use the column for categories and count for values
                value_counts = df[valid_columns[0]].value_counts().reset_index()
                value_counts.columns = [valid_columns[0], 'count']
                
                fig = px.bar(
                    value_counts,
                    x=valid_columns[0],
                    y='count',
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        elif chart_type == 'line':
            if len(valid_columns) >= 2:
                # Check if first column is datetime
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df[valid_columns[0]]):
                        df[valid_columns[0] + '_datetime'] = pd.to_datetime(df[valid_columns[0]], errors='coerce')
                        x_col = valid_columns[0] + '_datetime'
                    else:
                        x_col = valid_columns[0]
                except:
                    x_col = valid_columns[0]
                
                fig = px.line(
                    df,
                    x=x_col,
                    y=valid_columns[1],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
            elif len(valid_columns) == 1:
                # Create a line chart using index as x-axis
                df_temp = df.copy()
                df_temp['index'] = range(len(df_temp))
                
                fig = px.line(
                    df_temp,
                    x='index',
                    y=valid_columns[0],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                fig.update_layout(xaxis_title="Row Index", yaxis_title=valid_columns[0])
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        elif chart_type == 'scatter':
            if len(valid_columns) >= 2:
                config = {
                    'x': valid_columns[0],
                    'y': valid_columns[1],
                    'color': valid_columns[2] if len(valid_columns) > 2 else None
                }
                
                fig = px.scatter(
                    df,
                    x=config['x'],
                    y=config['y'],
                    color=config['color'],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        elif chart_type == 'pie':
            if valid_columns:
                # Count values for pie chart
                value_counts = df[valid_columns[0]].value_counts()
                
                # Limit to top categories if there are many
                if len(value_counts) > 15:
                    value_counts = value_counts.head(15)
                
                fig = px.pie(
                    names=value_counts.index,
                    values=value_counts.values,
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        elif chart_type == 'box' or chart_type == 'boxplot':
            if len(valid_columns) >= 2:
                config = {
                    'x': valid_columns[0],
                    'y': valid_columns[1]
                }
                
                fig = px.box(
                    df,
                    x=config['x'],
                    y=config['y'],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
            elif len(valid_columns) == 1:
                # Create a simple box plot for a single column
                fig = px.box(
                    df,
                    y=valid_columns[0],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        elif chart_type == 'histogram':
            if valid_columns:
                fig = px.histogram(
                    df,
                    x=valid_columns[0],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        elif chart_type == 'heatmap' or chart_type == 'correlation':
            if len(valid_columns) >= 2:
                # Create correlation matrix
                corr_df = df[valid_columns].corr()
                
                fig = px.imshow(
                    corr_df,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
            elif len(valid_columns) == 1:
                # Not enough columns for correlation
                return None, "Need at least 2 numeric columns for a correlation heatmap"
        elif chart_type == 'distribution':
            if valid_columns:
                # Create distribution plot
                fig = go.Figure()
                
                for col in valid_columns[:5]:  # Limit to 5 columns
                    # Ensure column is numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fig.add_trace(go.Histogram(
                            x=df[col],
                            name=col,
                            opacity=0.7,
                            histnorm='probability density'
                        ))
                
                fig.update_layout(
                    title=chart_idea['title'],
                    barmode='overlay',
                    template="plotly_white"
                )
                
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        
        # If we get here, try a generic approach based on column types
        if len(valid_columns) >= 1:
            # Check column types
            numeric_cols = [col for col in valid_columns if pd.api.types.is_numeric_dtype(df[col])]
            cat_cols = [col for col in valid_columns if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 10]
            
            if len(numeric_cols) >= 2:
                # Create scatter plot with two numeric columns
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
            elif len(numeric_cols) == 1 and len(cat_cols) >= 1:
                # Create bar chart with categorical and numeric
                fig = px.bar(
                    df,
                    x=cat_cols[0],
                    y=numeric_cols[0],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
            elif len(numeric_cols) == 1:
                # Create histogram with numeric column
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
            elif len(cat_cols) >= 1:
                # Create bar chart with counts of categorical column
                value_counts = df[cat_cols[0]].value_counts().reset_index()
                value_counts.columns = [cat_cols[0], 'count']
                
                fig = px.bar(
                    value_counts,
                    x=cat_cols[0],
                    y='count',
                    title=chart_idea['title'],
                    template="plotly_white"
                )
                return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), None
        
        # Default case: no valid visualization could be created
        return None, "Could not create a valid visualization for this chart type and data"
    except Exception as e:
        print(f"Error in create_llm_viz: {str(e)}")
        return None, str(e)

# Update the API route for creating LLM visualizations
@app.route('/api/create_llm_viz', methods=['POST'])
def create_llm_viz_route():
    if 'file_path' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    file_path = session.get('file_path')
    idea_index = request.json.get('idea_index', 0)
    
    try:
        # Convert string index to integer if needed
        if isinstance(idea_index, str):
            idea_index = int(idea_index)
        
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Create visualization
        viz_data, error = create_llm_viz(df, idea_index)
        
        if error:
            return jsonify({'error': error}), 400
        
        if viz_data:
            return jsonify({'success': True, 'data': viz_data})
        else:
            return jsonify({'error': 'Failed to create visualization'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/document-upload')
def document_upload():
    return render_template('document_upload.html')

# Add this to your imports if not already present
import tempfile

# Modify the upload-preview route to handle large files
@app.route('/upload-preview', methods=['POST'])
def upload_preview():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Check file extension
            if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
                return jsonify({'error': 'File type not supported. Please upload a CSV or Excel file.'}), 400
            
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file to a temporary file to handle large files
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
            file.save(temp_file.name)
            temp_file.close()
            
            # Generate a unique session ID for this dataset
            session_id = str(uuid.uuid4())
            
            # Load the data for preview - with chunking for large files
            try:
                if file.filename.endswith('.csv'):
                    # Use chunking for large CSV files
                    df = pd.read_csv(temp_file.name, nrows=10)  # Just read 10 rows for preview
                    total_rows = sum(1 for _ in open(temp_file.name, 'r')) - 1  # Count rows minus header
                else:
                    # For Excel, we'll use a different approach
                    df = pd.read_excel(temp_file.name, nrows=10)
                    xl = pd.ExcelFile(temp_file.name)
                    sheet_name = xl.sheet_names[0]  # Get first sheet
                    total_rows = xl.book.sheet_by_name(sheet_name).nrows - 1  # Count rows minus header
                
                # Store the file path in the session
                session['temp_file_path'] = temp_file.name
                session['session_id'] = session_id
                session['total_rows'] = total_rows
                
                # Create preview data
                preview_data = {
                    'columns': df.columns.tolist(),
                    'data': df.head(10).values.tolist()
                }
                
                return jsonify({
                    'success': True,
                    'preview': preview_data,
                    'session_id': session_id,
                    'total_rows': total_rows
                })
            except Exception as e:
                # Clean up temp file on error
                os.unlink(temp_file.name)
                return jsonify({'error': f'Error reading file: {str(e)}'}), 500
        except Exception as e:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File upload failed. Please try again.'}), 400

# Update the confirm-upload route to process the file with selected options
@app.route('/confirm-upload', methods=['POST'])
def confirm_upload():
    data = request.json
    session_id = data.get('sessionId')
    processing_options = data.get('processingOptions', {})
    
    if not session_id or session_id != session.get('session_id'):
        return jsonify({'error': 'Invalid session ID'}), 400
    
    file_path = session.get('temp_file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        # Load the data - with chunking for large files and handling encoding issues
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            # For large CSV files, use chunking with dask or process in batches
            if session.get('total_rows', 0) > 100000:  # Very large file
                # Use dask for large files if available, otherwise use chunking
                try:
                    import dask.dataframe as dd
                    df = dd.read_csv(file_path).compute()
                except ImportError:
                    # Fallback to chunking with pandas
                    chunk_size = 10000
                    chunks = []
                    
                    # Try different encodings if utf-8 fails
                    try:
                        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                            chunks.append(chunk)
                        df = pd.concat(chunks, ignore_index=True)
                    except UnicodeDecodeError:
                        # Try with different encodings
                        for encoding in ['latin1', 'iso-8859-1', 'cp1252']:
                            try:
                                chunks = []
                                for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding=encoding):
                                    chunks.append(chunk)
                                df = pd.concat(chunks, ignore_index=True)
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            return jsonify({'error': 'Unable to decode file with supported encodings. Please check the file encoding.'}), 400
            else:
                # Try different encodings if utf-8 fails
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    # Try with different encodings
                    for encoding in ['latin1', 'iso-8859-1', 'cp1252']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        return jsonify({'error': 'Unable to decode file with supported encodings. Please check the file encoding.'}), 400
        else:
            df = pd.read_excel(file_path)
        
        # Apply processing options
        if processing_options.get('handleMissingValues', True):
            # Handle missing values
            for col in df.columns:
                missing_pct = df[col].isna().mean() * 100
                if missing_pct > 0 and missing_pct < 30:  # If missing values are less than 30%
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Fill numeric columns with median
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        # Fill categorical columns with mode
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        df[col] = df[col].fillna(mode_value)
        
        if processing_options.get('convertDateColumns', True):
            # Convert date columns - but keep as date only (not datetime)
            for col in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        # Check if column might be a date
                        if df[col].dtype == 'object':
                            date_count = 0
                            for val in df[col].dropna().head(10):
                                try:
                                    pd.to_datetime(val)
                                    date_count += 1
                                except:
                                    pass
                            
                            # If most values look like dates, convert the column to date only (no time)
                            if date_count >= 7:  # If 7 out of 10 samples are dates
                                date_series = pd.to_datetime(df[col], errors='coerce')
                                # Convert to date only (no time component)
                                df[col] = date_series.dt.date
                except:
                    pass
        
        if processing_options.get('handleOutliers', True):
            # Handle outliers
            for col in df.select_dtypes(include=['number']).columns:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Save the processed data
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{os.path.basename(file_path)}")
        if processed_path.endswith('.csv'):
            df.to_csv(processed_path, index=False)
        else:
            df.to_excel(processed_path, index=False)
        
        # Clean up temp file
        try:
            os.unlink(file_path)
        except:
            pass
        
        # Store the processed file path in the session
        session['file_path'] = processed_path
        session['original_filename'] = os.path.basename(file_path)
        
        # Store basic info about the dataset
        session['rows'] = len(df)
        session['columns'] = len(df.columns)
        session['column_names'] = df.columns.tolist()
        
        # Analyze the data
        try:
            column_types = analyze_column_types(df)
            stats = get_column_stats(df)
            correlations = identify_correlations(df, column_types)
            potential_targets = identify_potential_targets(df, column_types)
            
            # Generate visualization recommendations
            recommendations = generate_visualization_recommendations(df, column_types, stats, correlations, potential_targets)
            
            # Generate LLM insights
            llm_insights = generate_llm_insights(df, column_types, stats, correlations, potential_targets)
            
            # Generate LLM chart ideas
            llm_chart_ideas = generate_llm_chart_ideas(df, column_types)
            
            # Store analysis results in session
            session['column_types'] = column_types
            session['stats'] = stats
            session['correlations'] = correlations
            session['potential_targets'] = potential_targets
            session['recommendations'] = recommendations
            session['llm_insights'] = llm_insights
            session['llm_chart_ideas'] = llm_chart_ideas
            
            # Create dataset info for chat
            df_info = f"""
            Dataset: {os.path.basename(file_path)}
            Rows: {len(df)}
            Columns: {len(df.columns)}
            Column types: {json.dumps(column_types)}
            """
            session['df_info'] = df_info
            
            # Add preprocessing info
            preprocessing_info = {
                "missing_values_handled": processing_options.get('handleMissingValues', True),
                "date_columns_converted": processing_options.get('convertDateColumns', True),
                "outliers_handled": processing_options.get('handleOutliers', True)
            }
            session['preprocessing_info'] = preprocessing_info
            
        except Exception as e:
            return jsonify({'error': f'Error analyzing data: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'redirect': url_for('overview')
        })
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/extract-tables-from-document', methods=['POST'])
def extract_tables_from_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Check file extension
            file_ext = file.filename.split('.')[-1].lower()
            if file_ext not in ['pdf', 'doc', 'docx', 'xls', 'xlsx']:
                return jsonify({'error': 'File type not supported. Please upload a PDF or Office document.'}), 400
            
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Generate a unique session ID for this extraction
            session_id = str(uuid.uuid4())
            
            # Extract tables based on file type
            tables = []
            
            if file_ext == 'pdf':
                # Extract tables from PDF
                pdf_tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
                
                # Convert to our format
                for i, table in enumerate(pdf_tables):
                    if not table.empty:
                        tables.append({
                            'title': f'Table {i+1}',
                            'headers': table.columns.tolist(),
                            'data': table.values.tolist(),
                            'sessionId': session_id
                        })
                
                # If tabula didn't find tables, try PyPDF2 for text extraction
                if not tables:
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        
                        # Simple table detection from text (very basic)
                        lines = text.split('\n')
                        table_data = []
                        in_table = False
                        
                        for line in lines:
                            if '|' in line or '\t' in line:
                                in_table = True
                                if '|' in line:
                                    row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                else:
                                    row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                                table_data.append(row)
                            elif in_table and line.strip() == '':
                                in_table = False
                                if table_data:
                                    headers = table_data[0] if len(table_data) > 0 else []
                                    data = table_data[1:] if len(table_data) > 1 else []
                                    tables.append({
                                        'title': 'Extracted Table',
                                        'headers': headers,
                                        'data': data,
                                        'sessionId': session_id
                                    })
                                table_data = []
            
            elif file_ext in ['doc', 'docx']:
                # Extract tables from Word document
                doc = docx.Document(file_path)
                
                for i, table in enumerate(doc.tables):
                    headers = []
                    data = []
                    
                    # Get headers from first row
                    if table.rows:
                        headers = [cell.text for cell in table.rows[0].cells]
                        
                        # Get data from remaining rows
                        for row in table.rows[1:]:
                            data.append([cell.text for cell in row.cells])
                    
                    tables.append({
                        'title': f'Table {i+1}',
                        'headers': headers,
                        'data': data,
                        'sessionId': session_id
                    })
            
            elif file_ext in ['xls', 'xlsx']:
                # Extract tables from Excel
                excel_file = pd.ExcelFile(file_path)
                
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if not df.empty:
                        tables.append({
                            'title': sheet_name,
                            'headers': df.columns.tolist(),
                            'data': df.values.tolist(),
                            'sessionId': session_id
                        })
            
            # Store extracted tables in session
            session['extracted_tables'] = tables
            session['document_path'] = file_path
            session['document_session_id'] = session_id
            
            if not tables:
                return jsonify({'error': 'No tables found in the document'}), 400
            
            return jsonify({
                'success': True,
                'tables': tables
            })
        except Exception as e:
            return jsonify({'error': f'Extraction failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File upload failed. Please try again.'}), 400

@app.route('/extract-tables-from-url', methods=['POST'])
def extract_tables_from_url():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        # Generate a unique session ID for this extraction
        session_id = str(uuid.uuid4())
        
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all tables
        html_tables = soup.find_all('table')
        
        if not html_tables:
            return jsonify({'error': 'No tables found on the webpage'}), 400
        
        # Extract tables
        tables = []
        
        for i, table in enumerate(html_tables):
            # Try to get table caption or title
            caption = table.find('caption')
            title = caption.get_text() if caption else f'Table {i+1}'
            
            # Get headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all('th')]
            
            if not headers and table.find('tr'):
                # Try to get headers from first row if thead not found
                first_row = table.find('tr')
                headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
            
            # Get data rows
            data = []
            body = table.find('tbody') or table
            rows = body.find_all('tr')
            
            # Skip the first row if we used it for headers
            start_idx = 1 if not header_row and headers else 0
            
            for row in rows[start_idx:]:
                cells = row.find_all(['td', 'th'])
                if cells:
                    data.append([cell.get_text().strip() for cell in cells])
            
            # Only add non-empty tables
            if headers or data:
                tables.append({
                    'title': title,
                    'headers': headers,
                    'data': data,
                    'sessionId': session_id
                })
        
        # Store extracted tables in session
        session['extracted_tables'] = tables
        session['url_source'] = url
        session['document_session_id'] = session_id
        
        return jsonify({
            'success': True,
            'tables': tables
        })
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500


@app.route('/process-selected-table', methods=['POST'])
def process_selected_table():
    data = request.json
    table_index = data.get('tableIndex')
    session_id = data.get('sessionId')
    
    if table_index is None:
        return jsonify({'error': 'No table selected'}), 400
    
    if not session_id or session_id != session.get('document_session_id'):
        return jsonify({'error': 'Invalid session ID'}), 400
    
    try:
        # Get the extracted tables
        tables = session.get('extracted_tables', [])
        
        if not tables or table_index >= len(tables):
            return jsonify({'error': 'Table not found'}), 400
        
        # Get the selected table
        selected_table = tables[table_index]
        
        # Convert to DataFrame
        df = pd.DataFrame(selected_table['data'], columns=selected_table['headers'])
        
        # Save to temporary CSV
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"extracted_table_{session_id}.csv")
        df.to_csv(temp_file, index=False)
        
        # Store the file path in the session
        session['file_path'] = temp_file
        session['original_file_path'] = session.get('document_path') or session.get('url_source')
        
        # Store basic info about the dataset
        session['rows'] = len(df)
        session['columns'] = len(df.columns)
        session['column_names'] = df.columns.tolist()
        
        # Analyze the data
        try:
            column_types = analyze_column_types(df)
            stats = get_column_stats(df)
            correlations = identify_correlations(df, column_types)
            potential_targets = identify_potential_targets(df, column_types)
            
            # Generate visualization recommendations
            recommendations = generate_visualization_recommendations(df, column_types, stats, correlations, potential_targets)
            
            # Generate LLM insights
            llm_insights = generate_llm_insights(df, column_types, stats, correlations, potential_targets)
            
            # Generate LLM chart ideas
            llm_chart_ideas = generate_llm_chart_ideas(df, column_types)
            
            # Store analysis results in session
            session['column_types'] = column_types
            session['stats'] = stats

        except Exception as e:
            return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

        return jsonify({
            'message': 'Table processed successfully',
            'filePath': temp_file,
            'rows': len(df),
            'columns': len(df.columns),
            'columnNames': df.columns.tolist(),
            'recommendations': recommendations,
            'llmInsights': llm_insights,
            'llmChartIdeas': llm_chart_ideas,
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Ensure debug mode is enabled during development for detailed error logs.
if __name__ == '__main__':
    app.run(debug=True)

