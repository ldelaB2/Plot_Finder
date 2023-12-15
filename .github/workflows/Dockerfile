# Use python as the base image
FROM python:3.9

# Install git
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Define Environment variables
ENV IRODS_HOST data.cyverse.org
ENV IRODS_PORT 1247
ENV IRODS_ZONE iplant
ENV IRODS_USER_NAME ""
ENV IRODS_PASSWORD ""
ENV IRODS_PATH ""

# Set the working directory
WORKDIR /app

# Clone GitHub repository and download dependencies
RUN git clone https://github.com/ldelaB2/Plot_Finder.git . && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install IRODS and make scripts executable
RUN mkdir -p /root/.irods/ && \
    chmod +x /app/install_iRODS.sh && \
    /app/install_iRODS.sh && \
    chmod +x /app/download_and_run.sh

# Specify the default command to run when the container starts
ENTRYPOINT ["/app/download_and_run.sh"]

CMD tail -f /dev/null
