# LLM Service Wrappers
# Lazy-loaded wrappers for Together AI functions from app.py

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.lazy_imports import get_pandas, get_together
from utils.memory_manager import cleanup_dataframe, MemoryMonitor
from utils.config import config

def generate_llm_insights_lazy(file_path: str) -> list:
    """
    Generate LLM insights with lazy loading and timeout
    
    Args:
        file_path: Path to data file
        
    Returns:
        list of insights
    """
    if not config.has_together_api_key:
        return ["API key required for LLM insights. Please set the TOGETHER_API_KEY environment variable."]
    
    with MemoryMonitor("LLM Insights"):
        pd = get_pandas()
        
        # Import original functions
        from original_app import (
            analyze_column_types,
            get_column_stats,
            identify_correlations,
            identify_potential_targets,
            generate_llm_insights
        )
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Limit data size for LLM processing
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)
            
            # Analyze data
            column_types = analyze_column_types(df)
            stats = get_column_stats(df)
            correlations = identify_correlations(df, column_types)
            potential_targets = identify_potential_targets(df, column_types)
            
            # Generate insights with timeout
            insights = generate_llm_insights(df, column_types, stats, correlations, potential_targets)
            
            cleanup_dataframe(df)
            
            return insights
            
        except Exception as e:
            return [f"Error generating insights: {str(e)}"]

def generate_llm_chart_ideas_lazy(file_path: str) -> list:
    """
    Generate LLM chart ideas with lazy loading
    
    Args:
        file_path: Path to data file
        
    Returns:
        list of chart ideas
    """
    if not config.has_together_api_key:
        return []
    
    with MemoryMonitor("LLM Chart Ideas"):
        pd = get_pandas()
        
        # Import original functions
        from original_app import analyze_column_types, generate_llm_chart_ideas
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Analyze column types
            column_types = analyze_column_types(df)
            
            # Generate chart ideas
            chart_ideas = generate_llm_chart_ideas(df, column_types)
            
            cleanup_dataframe(df)
            
            return chart_ideas
            
        except Exception as e:
            print(f"Error generating chart ideas: {str(e)}")
            return []

def chat_with_llm_lazy(question: str, df_info: str) -> str:
    """
    Chat with LLM about data with lazy loading
    
    Args:
        question: User question
        df_info: Dataset information
        
    Returns:
        LLM response
    """
    if not config.has_together_api_key:
        return "API key required for chat. Please set the TOGETHER_API_KEY environment variable."
    
    with MemoryMonitor("LLM Chat"):
        # Import original function
        from original_app import get_llm_response_for_chat
        
        try:
            response = get_llm_response_for_chat(question, df_info)
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"

def get_column_insights_lazy(file_path: str, column: str) -> list:
    """
    Get LLM insights for specific column with lazy loading
    
    Args:
        file_path: Path to data file
        column: Column name
        
    Returns:
        list of insights
    """
    if not config.has_together_api_key:
        return ["API key required for column insights."]
    
    with MemoryMonitor("Column Insights"):
        pd = get_pandas()
        
        # Import original functions
        from original_app import analyze_column_types, get_column_stats, generate_llm_column_insights
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            if column not in df.columns:
                return [f"Column '{column}' not found"]
            
            # Analyze data
            column_types = analyze_column_types(df)
            stats = get_column_stats(df)
            
            # Generate insights
            insights = generate_llm_column_insights(df, column, column_types[column], stats)
            
            cleanup_dataframe(df)
            
            return insights
            
        except Exception as e:
            return [f"Error generating column insights: {str(e)}"]
