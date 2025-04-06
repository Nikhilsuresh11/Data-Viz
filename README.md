# Data Visualization App

A Streamlit-based data visualization and analysis application that allows users to upload data files, explore datasets, create visualizations, and gain insights.

## Features

- **Data Upload**: Support for CSV and Excel files, URL imports, and sample datasets
- **Data Overview**: Quick summary of dataset characteristics
- **Data Explorer**: Detailed column analysis and filtering capabilities
- **Visualizations**: Automated visualization recommendations and custom chart builder
- **Insights**: Automated data insights and correlations
- **Custom Analysis**: Advanced filtering, custom charts, and statistical tests

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the application locally:

```bash
streamlit run streamlit_app.py
```

The app will be available at http://localhost:8501

## Deploying to Streamlit Cloud

This app is ready to be deployed to Streamlit Cloud:

1. Push to a GitHub repository
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select your repository
4. Configure deployment settings and deploy

## Data Processing

The app performs the following analysis on uploaded data:
- Column type detection (numeric, categorical, datetime, text)
- Basic statistics calculation
- Correlation analysis
- Visualization recommendations
- Automated insights generation

## Screenshots

[Screenshots will be added here]

## License

MIT License 