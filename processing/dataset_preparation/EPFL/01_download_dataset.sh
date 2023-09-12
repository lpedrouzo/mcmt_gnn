#!/bin/bash

# Check if the argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Directory path passed as an argument
target_directory="$1"

# Check if the target directory exists
if [ ! -d "$target_directory" ]; then
    echo "Directory not found: $target_directory"
    exit 1
fi

# Change to the target directory
cd "$target_directory"


mkdir -p datasets/EFPL/annotations
mkdir -p datasets/EFPL/embeddings
mkdir -p datasets/EFPL/videos

# Define an array of URLs
urls=(
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c0.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c1.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c2.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/6p-c3.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c0.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c1.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c2.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-c3.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c0.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c1.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c2.avi"
    "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video2/www/match5-c3.avi"
)

# Loop through the URLs and download each file
for url in "${urls[@]}"; do
    filename=$(basename "$url")
    wget "$url" -O "$filename"
done


# Initialize an empty array for videos
videos=()

# Find all .avi files in the current directory and add them to the videos array
for file in *.avi; do
    if [ -f "$file" ]; then
        videos+=("$file")
    fi
done

# Use awk to extract unique folder names based on the video filenames
folders=($(echo "${videos[@]}" | awk -F'-' '{print $1}' | tr ' ' '\n' | sort -u))

# Iterate over unique folders
for folder in "${folders[@]}"; do
    path="datasets/EFPL/videos/$folder/"
    
    # Check if the directory doesn't exist and create it
    if [ ! -d "$path" ]; then
        mkdir -p "$path"
    fi
    
    # Check if the directory is empty
    if [ -z "$(ls -A "$path")" ]; then
        # Move videos starting with the folder name to the target directory
        for video in "${videos[@]}"; do
            if [[ "$video" == "$folder"* ]]; then
                mv "$video" "$path"
            fi
        done
    fi
done

# List of directories to create or check for existence
directories=(
    "datasets/EFPL/videos/terrace"
    "datasets/EFPL/videos/basketball"
    "datasets/EFPL/videos/laboratory"
)

for directory in "${directories[@]}"; do
    if [ ! -d "$directory" ]; then
        mkdir -p "$directory"
        echo "Directory created: $directory"
    else
        echo "Directory already exists: $directory"
    fi
done

# Downloading ground truth files
git clone "https://bitbucket.org/merayxu/multiview-object-tracking-dataset.git"



# Define an array of source and destination folders
folders=(
    "Laboratory"
    "Basketball"
    "Terrace"
)

# Iterate through the folders and perform the move operations
for folder in "${folders[@]}"; do
    source_dir="multiview-object-tracking-dataset/EPFL/$folder datasets/EFPL/annotations"
    # Convert folder name to lowercase for the destination
    destination_dir="datasets/EFPL/annotations/$(echo "$folder" | tr '[:upper:]' '[:lower:]')"
    
    # Check if the source directory exists and the destination directory does not exist
    if [ -d "$source_dir" ] && [ ! -d "$destination_dir" ]; then
        mv "$source_dir" "$destination_dir"
        echo "Moved $source_dir to $destination_dir"
    fi
done

# Delete the parent folder (multiview-object-tracking-dataset/EPFL) regardless of whether it's empty or not
if [ -d "multiview-object-tracking-dataset" ]; then
    rm -r "multiview-object-tracking-dataset"
    echo "Deleted multiview-object-tracking-dataset"
fi