FROM python:3.9-slim
WORKDIR /app
RUN pip install torch numpy flask openai gunicorn
COPY . .
# Flask server ko background mein chalane ke liye
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "submission:app"]
