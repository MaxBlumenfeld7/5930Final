FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port for Gradio
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]