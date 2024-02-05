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

# Extract the img path
irdos_img_path=$(jq -r '.input_path' $(basename $IRODS_PATH))
irods_output_path=$(jq -r '.output_path' $(basename $IRODS_PATH))

# Downloading the image
iget -K -r -T --retries 5 -X input_checkpoint_file $irods_img_path .

# Update img_path and param file
local_img_path="$(pwd)/$(basename $img_path)"
local_output_path="$(pwd)/$(basename $irods_output_path)"

jq '.input_path = "'$local_img_path'" | .output_path = "'$local_output_path'"' $(basename $IRODS_PATH) > temp.json
mv temp.json $(basename $IRODS_PATH)

# Update the path to the params file
local_params_path="$(pwd)/$(basename $IRODS_PATH)"

# Running the script
python3 /app/scripts/main.py $local_params_path

# Pushing results to cyverse
#iput -K -f -b -r -T -f --retries 5 -X output_checkpoint_file  $IRODS_PATH

tail -f /dev/null
