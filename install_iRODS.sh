#!/bin/bash

# Install required packages
apt-get update
apt-get install -y wget lsb-release sudo

# Add the iRODS repository
wget -qO - https://packages.irods.org/irods-signing-key.asc | apt-key add -
echo "deb [arch=amd64] https://packages.irods.org/apt/ $(lsb_release -sc) main" \
  | tee /etc/apt/sources.list.d/renci-irods.list

# Update the package list
apt-get update

# Set pin preferences for iRODS packages
cat <<'EOF' | tee /etc/apt/preferences.d/irods
Package: irods-*
Pin: version 4.2.8
Pin-Priority: 1001
EOF

# Install iRODS iCommands
apt-get install -y irods-icommands

# Cleanup
apt-get clean
