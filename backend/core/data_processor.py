# Data Processing Wrappers
# Lazy-loaded wrappers for data analysis functions from app.py

import sys
import os

# Add parent directory to path to import from app.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.lazy_imports import get_pandas, get_numpy
from utils.memory_manager import cleanup_dataframe, MemoryMonitor, limit_dataframe_size
from utils.config import config

def analyze_data_lazy(file_path: str) -> dict:
    """
    Lazy-loaded data analysis wrapper
    
    Args:
        file_path: Path to data file
        
    Returns:
        dict with analysis results
    """
    with MemoryMonitor("Data Analysis"):
        # Lazy import pandas
        pd = get_pandas()
        
        # Import original functions from original_app.py
        from original_app import (
            analyze_column_types,
            get_column_stats,
            identify_correlations,
            identify_potential_targets
        )
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Limit size for memory safety
            df = limit_dataframe_size(df, config.MAX_ROWS_LIMIT)
            
            # Perform analysis using original functions
            column_types = analyze_column_types(df)
            stats = get_column_stats(df)
            correlations = identify_correlations(df, column_types)
            potential_targets = identify_potential_targets(df, column_types)
            
            result = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'column_types': column_types,
                'stats': stats,
                'correlations': {f"{k[0]}-{k[1]}": v for k, v in correlations.items()},
                'potential_targets': potential_targets
            }
            
            # Cleanup
            cleanup_dataframe(df)
            
            return result
            
        except Exception as e:
            raise Exception(f"Data analysis failed: {str(e)}")

def get_column_data_lazy(file_path: str, column: str) -> dict:
    """
    Get column-specific data with lazy loading
    
    Args:
        file_path: Path to data file
        column: Column name
        
    Returns:
        dict with column data
    """
    with MemoryMonitor("Column Data"):
        pd = get_pandas()
        np = get_numpy()
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found")
            
            # Get column data
            if pd.api.types.is_numeric_dtype(df[column]):
                hist, bin_edges = np.histogram(df[column].dropna(), bins=20)
                result = {
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
                }
            else:
                value_counts = df[column].value_counts().head(20)
                result = {
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
                }
            
            cleanup_dataframe(df)
            return result
            
        except Exception as e:
            raise Exception(f"Failed to get column data: {str(e)}")

def filter_data_lazy(file_path: str, filters: dict) -> dict:
    """
    Filter data with lazy loading
    
    Args:
        file_path: Path to data file
        filters: Filter criteria
        
    Returns:
        dict with filtered data info
    """
    with MemoryMonitor("Data Filtering"):
        pd = get_pandas()
        
        try:
            # Load data
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
                        filtered_df = filtered_df[
                            (filtered_df[col] >= filter_val['min']) & 
                            (filtered_df[col] <= filter_val['max'])
                        ]
                elif 'values' in filter_val and filter_val['values']:
                    filtered_df = filtered_df[filtered_df[col].isin(filter_val['values'])]
            
            result = {
                'rows': len(filtered_df),
                'sample': filtered_df.head(10).to_dict(orient='records')
            }
            
            cleanup_dataframe(df)
            cleanup_dataframe(filtered_df)
            
            return result
            
        except Exception as e:
            raise Exception(f"Data filtering failed: {str(e)}")
