#!/bin/bash

# Define Environment variables
export IRODS_ZONE='iplant'
export IRODS_PORT='1247'
export IRODS_HOST='data.cyverse.org'
export HOME='/app/.irods'

# Set up iRODS 
mkdir -p /app/.irods
touch /app/.irods/.irodsA

echo '{
  "irods_host": "'"$IRODS_HOST"'",
  "irods_port": '$IRODS_PORT',
  "irods_zone_name": "'"$IRODS_ZONE"'",
  "irods_user_name": "'"$IRODS_USER_NAME"'",
  "irods_authentication_scheme": "native",
  "irods_password": "'"$IRODS_PASSWORD"'"
}' > /app/.irods/irods_environment.json

# Set up iRODS environment variables
export IRODS_ENVIRONMENT_FILE=/app/.irods/irods_environment.json
export IRODS_AUTHENTICATION_FILE=/app/.irods/.irodsA

# Finalize iRODS connection
echo -e "$IRODS_PASSWORD\n" | iinit
export HOME='/'