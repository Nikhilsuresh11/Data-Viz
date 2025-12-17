# Backend - Lazy Loading Utilities
# This module provides lazy import functionality to reduce startup memory

import importlib
import sys
from typing import Any, Dict
from functools import lru_cache

class LazyImporter:
    """Lazy module importer to defer heavy imports until needed"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def import_module(self, module_name: str) -> Any:
        """Import module only when called, cache for reuse within request"""
        if module_name in self._cache:
            return self._cache[module_name]
        
        try:
            module = importlib.import_module(module_name)
            self._cache[module_name] = module
            return module
        except ImportError as e:
            raise ImportError(f"Failed to import {module_name}: {str(e)}")
    
    def clear_cache(self):
        """Clear import cache to free memory"""
        self._cache.clear()

# Global lazy importer instance (lightweight)
_lazy_importer = LazyImporter()

# Lazy import functions for heavy modules
def get_pandas():
    """Lazy load pandas"""
    return _lazy_importer.import_module('pandas')

def get_numpy():
    """Lazy load numpy"""
    return _lazy_importer.import_module('numpy')

def get_plotly_express():
    """Lazy load plotly.express"""
    return _lazy_importer.import_module('plotly.express')

def get_plotly_graph_objects():
    """Lazy load plotly.graph_objects"""
    return _lazy_importer.import_module('plotly.graph_objects')

def get_plotly_utils():
    """Lazy load plotly.utils"""
    return _lazy_importer.import_module('plotly.utils')

def get_matplotlib():
    """Lazy load matplotlib.pyplot"""
    return _lazy_importer.import_module('matplotlib.pyplot')

def get_seaborn():
    """Lazy load seaborn"""
    return _lazy_importer.import_module('seaborn')

def get_sklearn_preprocessing():
    """Lazy load sklearn.preprocessing"""
    return _lazy_importer.import_module('sklearn.preprocessing')

def get_together():
    """Lazy load together AI client"""
    return _lazy_importer.import_module('together')

def get_beautifulsoup():
    """Lazy load BeautifulSoup"""
    from bs4 import BeautifulSoup
    return BeautifulSoup

def get_tabula():
    """Lazy load tabula"""
    return _lazy_importer.import_module('tabula')

def get_pypdf2():
    """Lazy load PyPDF2"""
    return _lazy_importer.import_module('PyPDF2')

def get_docx():
    """Lazy load python-docx"""
    return _lazy_importer.import_module('docx')

def clear_lazy_imports():
    """Clear all lazy import caches"""
    _lazy_importer.clear_cache()
