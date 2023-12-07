# Use an official base image
FROM python:3.9

# Install git
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx

# Set the working directory
WORKDIR /app

# Clone your GitHub repository
RUN git clone https://github.com/ldelaB2/Plot_Finder.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Specify the default command to run when the container starts
ENTRYPOINT ["python", "/app/main.py"]
