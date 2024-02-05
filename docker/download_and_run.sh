#!/bin/bash

# Set up iRODS 
echo '{
  "irods_host": "'"$IRODS_HOST"'",
  "irods_port": '$IRODS_PORT',
  "irods_zone_name": "'"$IRODS_ZONE"'",
  "irods_user_name": "'"$IRODS_USER_NAME"'",
  "irods_authentication_scheme": "native",
  "irods_password": "'"$IRODS_PASSWORD"'"
}' > /root/.irods/irods_environment.json

# Finalize iRODS connection
echo -e "$IRODS_PASSWORD\n" | iinit

# Making working directory
mkdir -p /app/working_directory
cd /app/working_directory

# Downloading user specified params
iget -K -r -T --retries 5 -X input_checkpoint_file $IRODS_PATH .

# Defining input path
#LAST_FOLDER=$(basename $IRODS_PATH)
#SCRIPT_PATH="/app/working_directory/$LAST_FOLDER"

# Running the script
#python3 /app/scripts/main.py $SCRIPT_PATH

# Pushing results to cyverse
#cd $SCRIPT_PATH
#iput -K -f -b -r -T --retries 5 -X output_checkpoint_file Output $IRODS_PATH
