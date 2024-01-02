#!/bin/bash

# Define the path to the parent folder
parent_path="dataset/"

# Check if the parent path exists
if [ ! -d "$parent_path" ]; then
    echo "Error: The path $parent_path does not exist."
    exit 1
fi

# Get a list of all the PANCREAS_xxxx folders in the parent folder
folders=$(find $parent_path -type d -name 'PANCREAS_*')

# Loop over each folder and move the image.dcm files to the PANCREAS_xxxx folder
for folder in $folders; do
    echo "Moving files for $folder..."
    # Get a list of all the nested folders in the current folder
    nested_folders=$(find $folder -mindepth 1 -type d)

    # Loop over each nested folder and move the image.dcm files to the PANCREAS_xxxx folder
    for nested_folder in $nested_folders; do
        # Get a list of all the .dcm files in the nested folders and their subfolders
        dicom_files=$(find $nested_folder -type f -name '*.dcm')

        # Move each file to the destination folder
        for dicom_file in $dicom_files; do
            mv $dicom_file $folder/
        done

        # Remove the nested folders and their contents
        rm -r $nested_folder
    done
done
