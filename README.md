# NBA Player Points Prediction

## API Deployment
1. **Public Endpoint**: `http://127.0.0.1:8000/predict`
2. **Swagger UI**: `http://127.0.0.1:8000/docs`

## How to Run

### API Locally
```bash
cd api_service
pip install -r requirements.txt
uvicorn main:app --reload
