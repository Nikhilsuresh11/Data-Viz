# Optimized Flask Backend for Render Free Tier
# Lazy-loaded endpoints that call functions from ../app.py

from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import os
import sys
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Lightweight imports only
from utils.config import config
from utils.memory_manager import force_garbage_collection

# Initialize Flask app
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Session configuration for cross-origin requests (Vercel → Render)
app.config['SESSION_COOKIE_SAMESITE'] = 'None'  # Allow cross-site cookies
app.config['SESSION_COOKIE_SECURE'] = True      # Required for SameSite=None
app.config['SESSION_COOKIE_HTTPONLY'] = True    # Security best practice

# Enable CORS for Next.js frontend (development + production)
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    re.compile(r"^https://data-viz.*\.vercel\.app$"),  # Allow Vercel preview deployments
]
CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check for Render"""
    return jsonify({'status': 'healthy', 'service': 'data-viz-backend'}), 200

# File upload endpoint with lazy loading
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload and process file with lazy loading
    Lazy imports: pandas, file_handler
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Lazy import file handler
        from core.file_handler import process_uploaded_file_lazy
        
        # Get processing options
        processing_options = {
            'handleMissingValues': request.form.get('handleMissingValues', 'true').lower() == 'true',
            'convertDateColumns': request.form.get('convertDateColumns', 'true').lower() == 'true',
            'handleOutliers': request.form.get('handleOutliers', 'true').lower() == 'true'
        }
        
        # Process file
        result = process_uploaded_file_lazy(file, processing_options)
        
        # Store in session
        session['file_path'] = result['file_path']
        session['rows'] = result['rows']
        session['columns'] = result['columns']
        session['column_names'] = result['column_names']
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Data analysis endpoint with lazy loading
@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """
    Analyze uploaded data with lazy loading
    Lazy imports: pandas, numpy, data_processor
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded. Please upload a file first.'}), 400
    
    try:
        # Lazy import data processor
        from core.data_processor import analyze_data_lazy
        
        # Analyze data
        result = analyze_data_lazy(file_path)
        
        # Store analysis in session
        session['column_types'] = result['column_types']
        session['stats'] = result['stats']
        session['correlations'] = result['correlations']
        session['potential_targets'] = result['potential_targets']
        
        # Store dataset info for LLM context
        df_info = f"""Dataset Information:
- Rows: {result['rows']}
- Columns: {result['columns']}
- Column Names: {', '.join(result['column_names'])}
- Column Types: {', '.join([f"{col} ({dtype})" for col, dtype in result['column_types'].items()])}
"""
        session['df_info'] = df_info
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Visualization endpoint with lazy loading
@app.route('/api/visualize', methods=['POST'])
def create_visualization():
    """
    Create visualization with lazy loading
    Lazy imports: pandas, plotly, viz_generator
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.get_json()
    viz_type = data.get('type')
    viz_config = data.get('config', {})
    
    if not viz_type:
        return jsonify({'error': 'Visualization type required'}), 400
    
    try:
        # Lazy import viz generator
        from core.viz_generator import create_visualization_lazy
        
        # Create visualization
        result = create_visualization_lazy(file_path, viz_type, viz_config)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Visualization recommendations endpoint
@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """
    Get visualization recommendations with lazy loading
    Lazy imports: pandas, viz_generator
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        # Lazy import viz generator
        from core.viz_generator import generate_recommendations_lazy
        
        # Generate recommendations
        recommendations = generate_recommendations_lazy(file_path)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Custom chart endpoint
@app.route('/api/chart/custom', methods=['POST'])
def create_custom_chart():
    """
    Create custom chart with lazy loading
    Lazy imports: pandas, plotly, viz_generator
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.get_json()
    chart_type = data.get('chart_type')
    chart_config = data.get('config', {})
    
    if not chart_type:
        return jsonify({'error': 'Chart type required'}), 400
    
    try:
        # Lazy import viz generator
        from core.viz_generator import create_custom_chart_lazy
        
        # Create chart
        result = create_custom_chart_lazy(file_path, chart_type, chart_config)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# LLM insights endpoint
@app.route('/api/insights', methods=['GET'])
def get_insights():
    """
    Get LLM insights with lazy loading
    Lazy imports: pandas, together, llm_service
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        # Lazy import LLM service
        from core.llm_service import generate_llm_insights_lazy, generate_llm_chart_ideas_lazy
        
        # Generate insights
        insights = generate_llm_insights_lazy(file_path)
        chart_ideas = generate_llm_chart_ideas_lazy(file_path)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify({
            'insights': insights,
            'chart_ideas': chart_ideas
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with LLM about data with lazy loading
    Lazy imports: together, llm_service
    """
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question required'}), 400
    
    # Get dataset info from session
    df_info = session.get('df_info', 'No dataset information available')
    
    try:
        # Lazy import LLM service
        from core.llm_service import chat_with_llm_lazy
        
        # Get response
        response = chat_with_llm_lazy(question, df_info)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify({'response': response}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Column data endpoint
@app.route('/api/column/<column_name>', methods=['GET'])
def get_column_data(column_name):
    """
    Get column-specific data with lazy loading
    Lazy imports: pandas, numpy, data_processor
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        # Lazy import data processor
        from core.data_processor import get_column_data_lazy
        
        # Get column data
        result = get_column_data_lazy(file_path, column_name)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Column insights endpoint
@app.route('/api/column/<column_name>/insights', methods=['GET'])
def get_column_insights(column_name):
    """
    Get LLM insights for specific column with lazy loading
    Lazy imports: pandas, together, llm_service
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        # Lazy import LLM service
        from core.llm_service import get_column_insights_lazy
        
        # Get insights
        insights = get_column_insights_lazy(file_path, column_name)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify({'insights': insights}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Data filtering endpoint
@app.route('/api/filter', methods=['POST'])
def filter_data():
    """
    Filter data with lazy loading
    Lazy imports: pandas, data_processor
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.get_json()
    filters = data.get('filters', {})
    
    try:
        # Lazy import data processor
        from core.data_processor import filter_data_lazy
        
        # Filter data
        result = filter_data_lazy(file_path, filters)
        
        # Cleanup
        force_garbage_collection()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Export endpoint
@app.route('/api/export', methods=['POST'])
def export_data():
    """
    Export filtered data with lazy loading
    Lazy imports: pandas, file_handler
    """
    file_path = session.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.get_json()
    filters = data.get('filters')
    
    try:
        # Lazy import file handler
        from core.file_handler import export_filtered_data_lazy
        
        # Export data
        export_path = export_filtered_data_lazy(file_path, filters)
        
        # Send file
        response = send_file(
            export_path,
            as_attachment=True,
            download_name=os.path.basename(export_path),
            mimetype='text/csv'
        )
        
        # Cleanup
        force_garbage_collection()
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Session info endpoint
@app.route('/api/session', methods=['GET'])
def get_session_info():
    """Get current session information"""
    return jsonify({
        'has_data': 'file_path' in session and os.path.exists(session.get('file_path', '')),
        'rows': session.get('rows'),
        'columns': session.get('columns'),
        'column_names': session.get('column_names', [])
    }), 200

# Clear session endpoint
@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear session and cleanup"""
    session.clear()
    force_garbage_collection()
    return jsonify({'success': True}), 200

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=config.DEBUG)
