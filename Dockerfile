# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory (Avoid using root /)
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (model, app.py, etc.)
COPY . .

# EXPOSE the port for Hugging Face
EXPOSE 7860

# Start Streamlit and force it to use port 7860 on 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]