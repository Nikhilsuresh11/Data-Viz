# Memory Management Utilities
# Explicit cleanup and memory optimization for Render free tier

import gc
import sys
from typing import Any, Optional

def cleanup_dataframe(df: Any) -> None:
    """
    Explicitly delete DataFrame and free memory
    
    Args:
        df: pandas DataFrame to cleanup
    """
    if df is not None:
        try:
            del df
            gc.collect()
        except:
            pass

def cleanup_multiple(*objects) -> None:
    """
    Cleanup multiple objects at once
    
    Args:
        *objects: Variable number of objects to cleanup
    """
    for obj in objects:
        try:
            del obj
        except:
            pass
    gc.collect()

def get_memory_usage() -> dict:
    """
    Get current memory usage stats
    
    Returns:
        dict with memory info
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
    }

def force_garbage_collection() -> None:
    """Force garbage collection to free memory"""
    gc.collect()
    gc.collect()  # Call twice for better cleanup
    gc.collect()

def limit_dataframe_size(df: Any, max_rows: int = 100000) -> Any:
    """
    Limit DataFrame size to prevent memory issues
    
    Args:
        df: pandas DataFrame
        max_rows: Maximum number of rows to keep
        
    Returns:
        Limited DataFrame
    """
    if len(df) > max_rows:
        return df.head(max_rows)
    return df

def stream_dataframe_chunks(df: Any, chunk_size: int = 10000):
    """
    Generator to process DataFrame in chunks
    
    Args:
        df: pandas DataFrame
        chunk_size: Number of rows per chunk
        
    Yields:
        DataFrame chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]

class MemoryMonitor:
    """Context manager to monitor memory usage"""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_memory = None
        
    def __enter__(self):
        force_garbage_collection()
        try:
            self.start_memory = get_memory_usage()
            print(f"[{self.operation_name}] Start memory: {self.start_memory['rss_mb']:.2f} MB")
        except:
            pass
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            force_garbage_collection()
            end_memory = get_memory_usage()
            if self.start_memory:
                delta = end_memory['rss_mb'] - self.start_memory['rss_mb']
                print(f"[{self.operation_name}] End memory: {end_memory['rss_mb']:.2f} MB (Δ {delta:+.2f} MB)")
        except:
            pass
