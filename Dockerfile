# Use the official Python image from the Docker Hub
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME model_API

# Run app.py when the container launches
CMD ["python", "main.py"]
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
