# Use the official Python 3.11 slim image as a base image
FROM python:3.11-slim-buster
 
# Set the working directory inside the container to /app
WORKDIR /app
 
# Copy all files from the current directory (on your local system) to the /app directory inside the container
COPY . /app
 
# Update the package list for the system
#RUN apt update -y

RUN apt-get update && apt-get install -y libgomp1

COPY artifacts/ /app/artifacts/
 
# Install system-level dependencies and Python packages listed in requirements.txt
#RUN apt-get update && pip install -r requirements.txt
RUN pip install -r requirements.txt
 
# Expose the port that the application will run on (e.g., 5000)
EXPOSE 5000
 
# Set the default command to run when the container starts (running the app.py file with Python 3)
CMD ["python3", "app.py"]