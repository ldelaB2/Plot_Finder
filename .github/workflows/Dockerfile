# Use ubuntu as the base image
FROM ubuntu:22.04

# Install core dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y jq && \
    apt-get install -y nano && \
    # Verify Installation of python and pip
    python3 --version && pip3 --version && \
    # Clean up
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Clone GitHub repository and download dependencies
RUN git clone https://github.com/ldelaB2/Plot_Finder.git /app && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install IRODS and make scripts executable
RUN chmod +x /app/docker/install_iRODS.sh && \
    chmod +x /app/docker/download_and_run.sh && \
    chmod +x /app/docker/authenticate_irods.sh && \
    /app/docker/install_iRODS.sh && \
    chmod -R a+rwx /app

