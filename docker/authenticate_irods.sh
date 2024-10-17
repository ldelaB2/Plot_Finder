#!/bin/bash

# Set up iRODS 
mkdir -p /plot_finder/.irods
touch /plot_finder/.irods/.irodsA

echo '{
  "irods_host": "'"$IRODS_HOST"'",
  "irods_port": '$IRODS_PORT',
  "irods_zone_name": "'"$IRODS_ZONE"'",
  "irods_user_name": "'"$IRODS_USER_NAME"'",
  "irods_authentication_scheme": "native",
  "irods_password": "'"$IRODS_PASSWORD"'"
}' > /plot_finder/.irods/irods_environment.json

# Set up iRODS environment variables
export IRODS_ENVIRONMENT_FILE=/plot_finder/.irods/irods_environment.json
export IRODS_AUTHENTICATION_FILE=/plot_finder/.irods/.irodsA

# Finalize iRODS connection
echo -e "$IRODS_PASSWORD\n" | iinit