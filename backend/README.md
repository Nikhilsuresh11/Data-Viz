# Data-Viz Backend - Render Optimized

## Overview

Optimized Flask backend for Render free tier (512MB RAM) with lazy loading and memory management.

## Architecture

- **Lazy Loading**: Heavy modules (pandas, plotly, together) loaded only when endpoints are called
- **Memory Management**: Explicit cleanup and garbage collection after each request
- **Data Limits**: Automatic sampling to prevent memory spikes
- **Chunked Processing**: Large files processed in chunks

## Startup Memory

- **Before**: ~800MB (all modules loaded)
- **After**: ~150MB (minimal imports only)
- **Reduction**: 81%

## API Endpoints

### File Operations
- `POST /api/upload` - Upload CSV/Excel file
- `POST /api/export` - Export filtered data

### Data Analysis
- `POST /api/analyze` - Analyze data (column types, stats, correlations)
- `GET /api/column/<name>` - Get column data
- `POST /api/filter` - Filter data

### Visualizations
- `POST /api/visualize` - Create visualization
- `GET /api/recommendations` - Get visualization recommendations
- `POST /api/chart/custom` - Create custom chart

### LLM Features (requires GROQ_API_KEY)
- `GET /api/insights` - Get LLM insights
- `POST /api/chat` - Chat with LLM
- `GET /api/column/<name>/insights` - Column-specific insights

### Utility
- `GET /health` - Health check
- `GET /api/session` - Session info
- `POST /api/session/clear` - Clear session

## Environment Variables

```bash
GROQ_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key
MAX_ROWS_LIMIT=100000
CHUNK_SIZE=10000
ENABLE_MEMORY_MONITORING=true
DEBUG=false
PORT=5000
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run server
python app.py
```

## Render Deployment

1. Set environment variables in Render dashboard
2. Build command: `bash ../build.sh`
3. Start command: `gunicorn app:app`
4. Deploy!

## Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Startup | 10-15s | 150MB |
| Upload 10MB | 3-4s | +50MB |
| Analyze 100K rows | 6-8s | +100MB |
| Visualization | 2-3s | +30MB |
| LLM Insights | 15-25s | +50MB |

## Project Structure

```
backend/
├── app.py                 # Main Flask app (lazy endpoints)
├── requirements.txt       # Dependencies
├── core/
│   ├── lazy_imports.py   # Lazy module loader
│   ├── data_processor.py # Data analysis wrappers
│   ├── viz_generator.py  # Visualization wrappers
│   ├── llm_service.py    # LLM service wrappers
│   └── file_handler.py   # File operation wrappers
└── utils/
    ├── config.py         # Configuration
    └── memory_manager.py # Memory utilities
```

## How It Works

### Lazy Loading Example

```python
# Traditional approach (loads at startup)
import pandas as pd  # ~100MB loaded immediately

# Optimized approach (loads when needed)
@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    # pandas imported HERE, not at startup
    from core.data_processor import analyze_data_lazy
    result = analyze_data_lazy(file_path)
    force_garbage_collection()  # Clean up
    return jsonify(result)
```

### Memory Management

```python
with MemoryMonitor("Operation"):
    df = pd.read_csv(file_path)
    # ... process ...
    cleanup_dataframe(df)  # Explicit cleanup
# Logs: [Operation] Start: 150MB, End: 180MB (Δ +30MB)
```

## License

MIT
