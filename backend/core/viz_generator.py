# Visualization Generation Wrappers
# Lazy-loaded wrappers for chart generation from app.py

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.lazy_imports import get_pandas, get_plotly_express, get_plotly_graph_objects, get_plotly_utils
from utils.memory_manager import cleanup_dataframe, MemoryMonitor
from utils.config import config

def create_visualization_lazy(file_path: str, viz_type: str, viz_config: dict) -> dict:
    """
    Create visualization with lazy loading
    
    Args:
        file_path: Path to data file
        viz_type: Type of visualization
        viz_config: Visualization configuration
        
    Returns:
        dict with plotly figure JSON
    """
    with MemoryMonitor(f"Visualization: {viz_type}"):
        pd = get_pandas()
        
        # Import original visualization function
        from original_app import create_visualization
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Limit data points for visualization (max 10K for performance)
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
            
            # Create visualization using original function
            viz_data = create_visualization(df, viz_type, viz_config)
            
            cleanup_dataframe(df)
            
            if viz_data:
                return {'success': True, 'data': viz_data}
            else:
                return {'success': False, 'error': 'Failed to create visualization'}
                
        except Exception as e:
            raise Exception(f"Visualization creation failed: {str(e)}")

def generate_recommendations_lazy(file_path: str) -> list:
    """
    Generate visualization recommendations with lazy loading
    
    Args:
        file_path: Path to data file
        
    Returns:
        list of visualization recommendations
    """
    with MemoryMonitor("Viz Recommendations"):
        pd = get_pandas()
        
        # Import original functions
        from original_app import (
            analyze_column_types,
            get_column_stats,
            identify_correlations,
            identify_potential_targets,
            generate_visualization_recommendations
        )
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Analyze data
            column_types = analyze_column_types(df)
            stats = get_column_stats(df)
            correlations = identify_correlations(df, column_types)
            potential_targets = identify_potential_targets(df, column_types)
            
            # Generate recommendations
            recommendations = generate_visualization_recommendations(
                df, column_types, stats, correlations, potential_targets
            )
            
            cleanup_dataframe(df)
            
            return recommendations
            
        except Exception as e:
            raise Exception(f"Failed to generate recommendations: {str(e)}")

def create_custom_chart_lazy(file_path: str, chart_type: str, chart_config: dict) -> dict:
    """
    Create custom chart with lazy loading
    
    Args:
        file_path: Path to data file
        chart_type: Type of chart (bar, line, scatter, etc.)
        chart_config: Chart configuration
        
    Returns:
        dict with plotly figure JSON
    """
    with MemoryMonitor(f"Custom Chart: {chart_type}"):
        pd = get_pandas()
        px = get_plotly_express()
        go = get_plotly_graph_objects()
        plotly_utils = get_plotly_utils()
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Limit data points
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
            
            fig = None
            
            # Create chart based on type
            if chart_type == 'bar':
                x_col = chart_config.get('x')
                y_col = chart_config.get('y')
                color_col = chart_config.get('color')
                
                if not x_col or not y_col:
                    raise ValueError('Missing required columns for bar chart')
                    
                fig = px.bar(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col if color_col != 'None' else None,
                    title=f"{y_col} by {x_col}",
                    template="plotly_white"
                )
                
            elif chart_type == 'line':
                x_col = chart_config.get('x')
                y_cols = chart_config.get('y', [])
                
                if not x_col or not y_cols:
                    raise ValueError('Missing required columns for line chart')
                    
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
                x_col = chart_config.get('x')
                y_col = chart_config.get('y')
                color_col = chart_config.get('color')
                
                if not x_col or not y_col:
                    raise ValueError('Missing required columns for scatter plot')
                    
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col if color_col != 'None' else None,
                    title=f"Scatter Plot: {y_col} vs {x_col}",
                    template="plotly_white"
                )
                
            elif chart_type == 'histogram':
                col = chart_config.get('column')
                bins = chart_config.get('bins', 30)
                
                if not col:
                    raise ValueError('Missing required column for histogram')
                    
                fig = px.histogram(
                    df, 
                    x=col,
                    nbins=bins,
                    title=f"Histogram of {col}",
                    template="plotly_white"
                )
            
            if fig:
                fig_json = json.loads(json.dumps(fig, cls=plotly_utils.PlotlyJSONEncoder))
                cleanup_dataframe(df)
                return {'success': True, 'data': fig_json}
            else:
                cleanup_dataframe(df)
                return {'success': False, 'error': 'Unsupported chart type'}
                
        except Exception as e:
            raise Exception(f"Custom chart creation failed: {str(e)}")
