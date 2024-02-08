# Use ubuntu as the base image
FROM ubuntu:22.04

# Install core dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y jq && \
    # Verify Installation of python and pip
    python3 --version && pip3 --version && \
    # Clean up
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Define Environment variables
ENV IRODS_HOST data.cyverse.org
ENV IRODS_PORT 1247
ENV IRODS_ZONE iplant


# Set the working directory
WORKDIR /app

# Clone GitHub repository and download dependencies
RUN git clone https://github.com/ldelaB2/Plot_Finder.git . && \
    pip install --no-cache-dir -r /app/requirements.txt

# Create new user and set premission
RUN useradd -ms /bin/bash appuser && \
    chown -R appuser /app
    #chown appuser /root/.irods/irods_environment.json && \
    #chmod -R o+rx /root/.irods

# Install IRODS and make scripts executable
RUN mkdir -p /home/appuser/.irods && \
    touch /home/appuser/.irods/irods_environment.json && \
    chown appuser /home/appuser/.irods/irods_environment.json && \
    chmod +x /app/docker/install_iRODS.sh && \
    /app/docker/install_iRODS.sh && \
    chmod +x /app/docker/download_and_run.sh && \
    mkdir -p /app/working_directory




# Specify the default command to run when the container starts
#ENTRYPOINT ["/bin/bash"]
