#!/bin/bash

##### Figure out which source we are using ######
# Data set path
data_set_path="/plot_finder/data_set"
docker_param="docker_source"

# Check if the param.json file exists, if it does read it in
param_file="${data_set_path}/param.json"

if [[ -f "$param_file" ]]; then
    echo "Param File Found"

    # Check if the docker source exists and read its value
    if jq -e ".$docker_param" "$param_file" > /dev/null; then
        docker_source=$(jq -r ".$docker_param" "$param_file")
    else
        echo "'$docker_param' not found in the param file, exiting ..."
        exit 1
    fi

else
    echo "Param file not found, please double check directory/ naming."
    exit 1
fi

#### Download from IRODS if needed ####
if [[ "$docker_source" == "cyverse" ]]; then
    echo "Using Cyverse Source"
    export IRODS_HOST=$(jq -r '.IRODS_HOST' "$param_file")
    export IRODS_PORT=$(jq -r '.IRODS_PORT' "$param_file")
    export IRODS_ZONE=$(jq -r '.IRODS_ZONE' "$param_file")
    export IRODS_USER_NAME=$(jq -r '.IRODS_USER_NAME' "$param_file")
    export IRODS_PASSWORD=$(jq -r '.IRODS_PASSWORD' "$param_file")

    source /plot_finder/docker/authenticate_irods.sh
    echo "Completed Authentication with Cyverse"


# Check the docker source
elif [[ "$docker_source" == "local" ]]; then
    echo "Using Local Source"

else
    echo "Unknown Docker Source"
    exit 1
fi

# Check that the image exists
image_file=$(find "$data_set_path" -type f \( -iname "*.tif" -o -iname "*.tiff" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \))

if [[ -f "$image_file" ]]; then
    echo "Image File Found at: $image_file"
    # Update the param file with the image path
    jq '(.ortho_path) = "'$image_file'"' "$param_file" > "$data_set_path/final_params.json"

else
    echo "Image file not found, please double check directory/naming."
    exit 1
fi


# Run the plot finder
python3 /plot_finder/scripts/main.py "$data_set_path/final_params.json"
