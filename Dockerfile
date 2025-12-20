FROM python:3.12.7-slim
WORKDIR /app
COPY MLProject/ /app/
RUN pip install --no-cache-dir pandas scikit-learn mlflow dagshub
CMD ["python", "modelling.py"]
