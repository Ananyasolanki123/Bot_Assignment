FROM python:3.12

# Set working directory inside container
WORKDIR /app

COPY requirement.txt ./requirement.txt

RUN pip install --no-cache-dir -r requirement.txt

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

