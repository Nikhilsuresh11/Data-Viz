# File Handler Wrappers
# Lazy-loaded wrappers for file operations from app.py

import sys
import os
import tempfile
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.lazy_imports import get_pandas
from utils.memory_manager import cleanup_dataframe, MemoryMonitor
from utils.config import config

def process_uploaded_file_lazy(file, processing_options: dict = None) -> dict:
    """
    Process uploaded file with lazy loading and chunking
    
    Args:
        file: Uploaded file object
        processing_options: Processing configuration
        
    Returns:
        dict with file info and processed path
    """
    if processing_options is None:
        processing_options = {
            'handleMissingValues': True,
            'convertDateColumns': True,
            'handleOutliers': True
        }
    
    with MemoryMonitor("File Upload"):
        pd = get_pandas()
        
        try:
            # Secure filename
            filename = secure_filename(file.filename)
            
            # Check file type
            if not (filename.endswith('.csv') or filename.endswith('.xlsx')):
                raise ValueError('File type not supported. Please upload a CSV or Excel file.')
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            file.save(temp_file.name)
            temp_file.close()
            
            # Load data with chunking for large files
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.csv':
                # Try to read first to check size
                try:
                    df = pd.read_csv(temp_file.name, nrows=10)
                    total_rows = sum(1 for _ in open(temp_file.name, 'r')) - 1
                except:
                    total_rows = 0
                
                # If large file, process in chunks
                if total_rows > config.MAX_ROWS_LIMIT:
                    df = pd.read_csv(temp_file.name, nrows=config.MAX_ROWS_LIMIT)
                else:
                    df = pd.read_csv(temp_file.name)
            else:
                df = pd.read_excel(temp_file.name)
            
            # Apply processing options (simplified from app.py)
            if processing_options.get('handleMissingValues', True):
                for col in df.columns:
                    missing_pct = df[col].isna().mean() * 100
                    if 0 < missing_pct < 30:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                            df[col] = df[col].fillna(mode_value)
            
            # Save processed file
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(config.UPLOAD_FOLDER, processed_filename)
            
            if processed_path.endswith('.csv'):
                df.to_csv(processed_path, index=False)
            else:
                df.to_excel(processed_path, index=False)
            
            result = {
                'success': True,
                'file_path': processed_path,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
            # Cleanup
            cleanup_dataframe(df)
            try:
                os.unlink(temp_file.name)
            except:
                pass
            
            return result
            
        except Exception as e:
            raise Exception(f"File processing failed: {str(e)}")

def export_filtered_data_lazy(file_path: str, filters: dict = None) -> str:
    """
    Export filtered data to CSV
    
    Args:
        file_path: Path to data file
        filters: Filter criteria
        
    Returns:
        Path to exported file
    """
    with MemoryMonitor("Data Export"):
        pd = get_pandas()
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Apply filters if provided
            if filters:
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
            else:
                filtered_df = df
            
            # Export to temp file
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(config.UPLOAD_FOLDER, f"export_{timestamp}.csv")
            filtered_df.to_csv(export_path, index=False)
            
            cleanup_dataframe(df)
            cleanup_dataframe(filtered_df)
            
            return export_path
            
        except Exception as e:
            raise Exception(f"Data export failed: {str(e)}")
