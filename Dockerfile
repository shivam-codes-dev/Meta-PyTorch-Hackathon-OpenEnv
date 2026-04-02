FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install torch numpy openai

COPY . .

CMD ["python", "inference.py"]
