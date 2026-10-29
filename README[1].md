# ML Model Deployment Strategies and Model Serving using FastAPI

## 1. Project Overview

This project demonstrates machine learning model training, model saving, online prediction, batch prediction, streaming simulation, and API deployment using FastAPI.

## 2. Dataset

Dataset: Kaggle Titanic train.csv

Target column: Survived

- 0: Did not survive
- 1: Survived

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Google Colab

## 4. Model Training

A Logistic Regression classifier was trained using a Scikit-learn pipeline. The pipeline includes missing-value handling, numerical scaling, categorical encoding, and classification.

## 5. Model Evaluation

The model was evaluated using a separate test dataset.

Accuracy: 73.88%

## 6. Model Saving

The complete trained pipeline was saved as model.pkl and reused for predictions without retraining.

## 7. Prediction Strategies

### Online Prediction
Predicts one unseen passenger record using the saved model.

### Batch Prediction
Predicts 20 unseen passenger records together. Results are saved in batch_predictions.csv.

### Streaming Simulation
Processes 10 unseen passenger records individually with a one-second delay between predictions.

## 8. FastAPI Endpoints

- GET /health: Checks API health.
- POST /predict: Predicts passenger survival.

Swagger UI is available at /docs when the API server is running.

## 9. Input Validation

Pydantic validates passenger input. Valid requests return HTTP 200. An invalid passenger class, such as Pclass = 5, returns HTTP 422.

## 10. Project Files

- app.py: FastAPI application
- client.py: Python API client
- model.pkl: Saved trained ML pipeline
- batch_predictions.csv: Batch prediction results
- observations.pdf: Assignment observations
- requirements.txt: Required Python packages

## 11. Conclusion

This project demonstrates a reusable machine learning pipeline and three prediction strategies, along with REST API deployment and input validation.
