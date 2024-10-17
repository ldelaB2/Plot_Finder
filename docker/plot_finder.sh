#!/bin/bash

##### Figure out which source we are using ######
# Data set path
data_set_path="/data_set"
docker_param="docker_source"

# Check if the param.json file exists, if it does read it in
param_file="${data_set_path}/param.json"

if [[ -f "$param_file" ]]; then
    echo "Param File Found"

    # Check if the docker source exists and read its value
    if jq -e ".$docker_param" "$param_file" > /dev/null; then
        docker_source=$(jq -r ".$docker_param" "$param_file")
        echo "Using Docker Source: $docker_source"
    else
        echo "'$docker_param' not found in the param file, exiting ..."
        exit 1
    fi

else
    echo "Param file not found, please double check directory/ naming."
    exit 1
fi

#### Local Source ###

if [[ "$docker_source" == "local" ]]; then
    echo "Using Local Source"

else
    echo "Using IRODS"