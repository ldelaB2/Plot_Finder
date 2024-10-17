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
RUN git clone https://github.com/ldelaB2/Plot_Finder.git /plot_finder && \
    pip install --no-cache-dir -r /plot_finder/requirements.txt

# Install IRODS and make scripts executable
RUN chmod +x /plot_finder/docker/install_iRODS.sh && \
    chmod +x /plot_finder/docker/download_from_cyverse.sh && \
    chmod +x /plot_finder/docker/authenticate_irods.sh && \
    /plot_finder/docker/install_iRODS.sh && \
    chmod -R a+rwx /plot_finder

