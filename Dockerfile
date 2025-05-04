# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the local project files into the container
COPY . ./
COPY text_chunks.pkl /app/text_chunks.pkl
COPY tfidf_vectors.pkl /app/tfidf_vectors.pkl
COPY vectorizer.pkl /app/vectorizer.pkl
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Cloud Run expects
#EXPOSE 8085

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false"]
