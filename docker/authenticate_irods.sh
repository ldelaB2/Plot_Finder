#!/bin/bash

# Define Environment variables
export IRODS_ZONE='iplant'
export IRODS_PORT='1247'
export IRODS_HOST='data.cyverse.org'

# Set up iRODS 
cd /app
mkdir -p $PWD/.irods

echo '{
  "irods_host": "'"$IRODS_HOST"'",
  "irods_port": '$IRODS_PORT',
  "irods_zone_name": "'"$IRODS_ZONE"'",
  "irods_user_name": "'"$IRODS_USER_NAME"'",
  "irods_authentication_scheme": "native",
  "irods_password": "'"$IRODS_PASSWORD"'"
}' > $PWD/.irods/irods_environment.json

# Set up iRODS environment variables
export IRODS_ENVIRONMENT_FILE=$PWD/.irods/irods_environment.json
touch $PWD/.irods/.irodsA
export IRODS_AUTHENTICATION_FILE=$PWD/.irods/.irodsA

# Finalize iRODS connection
echo -e "$IRODS_PASSWORD\n" | iinit