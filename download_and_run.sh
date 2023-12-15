#!/bin/bash

# Making working directory
mkdir -p /app/working_directory
cd /app/working_directory

# Downloading images
iget -K -r -T --retries 5 -X input_checkpoint_file $IRODS_PATH .

# Defining input path
LAST_FOLDER=$(basename $IRODS_PATH)
SCRIPT_PATH="/app/working_directory/$LAST_FOLDER"

# Running the script
python3 /app/main.py $SCRIPT_PATH

# Pushing results to cyverse
RESULTS_PATH="$SCRIPT_PATH/Output"
iput -K -f -b -r -T --retries 5 -X output_checkpoint_file $RESULTS_PATH $IRODS_PATH

