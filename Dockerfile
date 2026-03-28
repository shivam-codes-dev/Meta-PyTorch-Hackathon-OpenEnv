FROM python:3.9-slim

WORKDIR /app

# Sabse pehle basic requirements install karein
RUN pip install --no-cache-dir torch numpy

# Saari files copy karein
COPY . .

# Scaler system ko batayein ki inference.py main entry point hai
CMD ["python", "inference.py"]
