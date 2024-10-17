#!/bin/bash

# Set up the working directory
mkdir -p /app/working_directory
cd /app/working_directory

# Downloading user specified params
iget -K -r -T --retries 5 -X input_checkpoint_file $IRODS_PATH .

# Extract the img path
irods_img_path=$(jq -r '.input_path' $(basename $IRODS_PATH))
irods_output_path=$(jq -r '.output_path' $(basename $IRODS_PATH))

# Downloading the image


# Update img_path and param file
local_img_path="$(pwd)/$(basename $irods_img_path)"
local_output_path="$(pwd)/$(basename $irods_output_path)"

jq '.input_path = "'$local_img_path'" | .output_path = "'$local_output_path'"' $(basename $IRODS_PATH) > temp.json
mv temp.json $(basename $IRODS_PATH)

# Update the path to the params file
local_params_path="$(pwd)/$(basename $IRODS_PATH)"

# Move default params to the correct location
cp /app/scripts/default_params.json /app/working_directory/default_params.json

# Running the script
python3 /app/scripts/main.py $local_params_path

# Pushing results to cyverse
output_checkpoint_file="$(pwd)/output_checkpoint_file"
cd $local_output_path
iput -K -f -b -r -T --retries 5 -X $output_checkpoint_file . $irods_output_path

