FROM python:3.9-slim
WORKDIR /app
RUN pip install torch numpy
COPY . .
CMD ["python", "inference.py"]
