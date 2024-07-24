#!/usr/bin/env bash

export ID=$1
export NAME=$2
export PROJ=$3

# set up output logging file
if [ ! -d "/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/" ]; then mkdir "/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/"; fi
output_file="/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/run_download_and_unzip_${ID}_$(date +'%Y%m%d_%H%M%S').txt"
exec > "$output_file" 2>&1

echo
echo "Execute bash script."
echo
start_time=$(date +%s)

# test/report environment variables
echo ID = $ID
echo NAME = $NAME
echo PROJ = $PROJ

# activate virtual environment
source /arc/projects/salvage/environments/salvage_env/bin/activate

# if data already exists, delete it
echo "Deleting data (if it already exists)..."
echo "rm -rf /arc/projects/salvage/ALMA_data/$ID/$PROJ/"
rm -rf "/arc/projects/salvage/ALMA_data/${ID}/${PROJ}/*"
rm -rvf "/arc/projects/salvage/ALMA_data/${ID}/${PROJ}/"
echo "rm -rf /arc/projects/salvage/ALMA_data/$ID/*.pickle"
rm -rvf "/arc/projects/salvage/ALMA_data/$ID/*.pickle"
sleep 5

# check if final location for data (indexed by objID) exists
# if not, create that directory
data_dir="/arc/projects/salvage/ALMA_data/$ID/"

if [ ! -d "$data_dir" ]; then
    echo "Directory for this SDSS object does not exist. Creating $data_dir ..."
    mkdir "$data_dir"
else
    echo "Directory for this SDSS object already exists."
fi

# run script to download data into temporary directory
echo "Downloading ALMA data..."
cd /arc/projects/salvage/ALMA_reduction/py_scripts/
python download_alma_data.py
sleep 5

# checking if there was a connection error and re-running if there was ....
if ls $data_dir*.tar 1> /dev/null 2>&1; then
    echo "Script downloaded at least one tar file. Carry on."
else
    echo "No .tar files found in $data_dir"
    echo "I suspect there was a connection error with the archive. Try again."
    echo "Downloading ALMA data..."
    python download_alma_data.py
fi
sleep 5

# unpack the data into final location
echo "Unzipping downloaded data..."
cd "$data_dir"
for tar_file in *.tar; do
    # Check if the file is a regular file
    if [ -f "$tar_file" ]; then
        # Extract the contents of the .tar file into final location
        tar -xf "$tar_file" #-C "$data_dir"
        echo "Extracted $tar_file into $data_dir"
    fi
done
sleep 5

#echo "Deleting temporary directory and data therein ..."
#cd ..
#rm -rvf "$tmp_dir"

# Record the end time and print execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Script execution time: $duration seconds"

# make completion file (allows for running in serial)
touch "/arc/projects/salvage/ALMA_reduction/salvage_completion_files/${ID}_download_complete.txt"

echo
echo "Bash script complete."
echo

# make completion file (allows for running in serial)
touch "/arc/projects/salvage/ALMA_reduction/salvage_completion_files/${ID}_download_complete.txt"

deactivate
