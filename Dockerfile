FROM python:3.12

# Set working directory inside container
WORKDIR /app

# Copy requirements first for better caching
COPY requirement.txt ./requirement.txt

# Install python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the full project into the container
COPY . .

# Ensure src is discoverable for imports
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
