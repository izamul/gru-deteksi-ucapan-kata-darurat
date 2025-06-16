# Flask + Streamlit Integration Demo

This project demonstrates the integration of Flask (backend API) and Streamlit (frontend interface).

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

You'll need to run both the Flask backend and Streamlit frontend in separate terminal windows.

### Terminal 1 - Flask Backend
```bash
python app.py
```
The Flask API will run on http://localhost:5000

### Terminal 2 - Streamlit Frontend
```bash
streamlit run streamlit_app.py
```
The Streamlit interface will be available at http://localhost:8501

## Features

- Flask RESTful API endpoints
- Interactive Streamlit UI
- Real-time API communication
- Responsive design
- Navigation sidebar

## API Endpoints

- `GET /`: Welcome message
- `GET /api/hello`: Hello message endpoint

## Requirements

- Python 3.7+
- Flask
- Streamlit
- requests 