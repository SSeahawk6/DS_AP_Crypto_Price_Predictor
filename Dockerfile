# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
# We verify connectivity and upgrade pip first
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Default command: Runs the full comparison pipeline for all coins
CMD ["python", "main.py", "--coins", "all", "--days", "1825"]
