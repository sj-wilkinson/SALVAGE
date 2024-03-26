#!/usr/bin/env bash

export ID=$1
export NAME=$2

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

#conda activate almaredux
#pip install alminer

# check if final location for data (indexed by objID) exists
# if not, create that directory
data_dir="/arc/projects/salvage/ALMA_data/$ID"

if [ ! -d "$data_dir" ]; then
    echo "Directory for data does not exist. Creating $data_dir ..."
    mkdir $data_dir
else
    echo "Directory for data already exists."
fi

# if data already exists, delete it
echo "Deleting data (if it already exists)..."
cd $data_dir
rm -rfv *.tar
sleep 5

# create temporary sub-directory for this specific ALMA source name
# to avoid interference with other ALMA projects observing this SDSS target
tmp_dir="/arc/projects/salvage/ALMA_data/$ID/$NAME/"

# check if the temporary directory exists (it shouldn't)
# if it does, delete it and re-make, if it doesn't make it
if [ ! -d "$tmp_dir" ]; then
    echo "Directory for data does not exist. Creating $tmp_dir ..."
    mkdir $tmp_dir
else
    echo "Directory for data already exists. Re-creating ..."
    rm -rvf $tmp_dir
    mkdir $tmp_dir
    
fi

echo "Downloading ALMA data..."
cd /arc/projects/salvage/ALMA_reduction/py_scripts/
python download_alma_data.py
sleep 5

echo "Unzipping downloaded data..."
cd $tmp_dir
for tar_file in *.tar; do
    # Check if the file is a regular file
    if [ -f "$tar_file" ]; then
        # Extract the contents of the .tar file into final location
        tar -xf "$tar_file" -C $data_dir
        echo "Extracted: $tar_file"
    fi
done
sleep 5

echo "Deleting temporary directory and data therein ..."
cd ..
rm -rvf $tmp_dir

# Record the end time and print execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Script execution time: $duration seconds"

echo
echo "Bash script complete."
echo