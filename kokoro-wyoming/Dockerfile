FROM python:3.12-slim
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
ENV PYTHONPATH=/app
WORKDIR /app/src
# TODO: figure out how to use the DEBUG env variable instead
CMD ["python", "main.py", "--debug"]