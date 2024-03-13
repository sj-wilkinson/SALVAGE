#!/usr/bin/env bash

export ID=$1
export NAME=$2
#export TAR1=$3
#export TAR2=$4
#export ASDM=$5

# set up output logging file
output_file="/arc/projects/salvage/ALMA_reduction/logs/download_and_unzip_${ID}_$(date +'%Y%m%d_%H%M%S').txt"
exec > "$output_file" 2>&1

echo
echo "Execute bash script."
echo
start_time=$(date +%s)

# test/report environment variables
echo ID = $ID
echo NAME = $NAME

#conda activate almaredux
#pip install alminer

# check if appropriate data directory exists
data_dir="/arc/projects/salvage/ALMA_data/$ID"

if [ ! -d "$data_dir" ]; then
    echo "Directory for data does not exist. Creating $data_dir ..."
    mkdir $data_dir
else
    echo "Directory for data already exists."
fi

echo "Deleting data (if it already exists)..."
cd $data_dir
rm -rfv *.tar
sleep 5

echo "Downloading ALMA data..."
cd /arc/projects/salvage/ALMA_reduction/py_scripts/
python download_alma_data.py
sleep 5

echo "Unzipping downloaded data..."
cd $data_dir
for tar_file in *.tar; do
    # Check if the file is a regular file
    if [ -f "$tar_file" ]; then
        # Extract the contents of the .tar file
        tar -xf "$tar_file"
        echo "Extracted: $tar_file"
    fi
done
sleep 5

echo "Deleting data..."
rm -rfv *.tar
rm -rfv *.pickle

# Record the end time and print execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Script execution time: $duration seconds"

echo
echo "Bash script complete."
echo