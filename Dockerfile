# Use Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
# Copy MLflow runs if needed for the demo
# COPY mlruns/ ./mlruns/

# Set identification environment variables
ENV STUDENT_NAME="Sumedh Kolupoti"
ENV ROLL_NO="2022BCS0169"

# Expose port
EXPOSE 8000

# Run terminal command
CMD ["uvicorn", "src.api_2022BCS0169:app", "--host", "0.0.0.0", "--port", "8000"]
