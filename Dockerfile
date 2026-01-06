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

# Default command: Runs the analysis for Bitcoin (fast check)
# User can override this (e.g., docker run my-image python main.py --coins all)
CMD ["python", "main.py", "--coins", "bitcoin"]
