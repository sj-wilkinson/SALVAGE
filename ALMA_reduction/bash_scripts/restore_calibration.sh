#!/usr/bin/env bash

source ~/.bashrc

export ID=$1
export PROJ=$2
export UPATH=$3

# set up output logging file
output_file="/arc/projects/salvage/ALMA_reduction/logs/restore_calibration_${ID}_$(date +'%Y%m%d_%H%M%S').txt"
exec > "$output_file" 2>&1

echo
echo "Execute bash script."
echo
start_time=$(date +%s)

# test/report environment variables
echo ID = $ID
echo PROJ = $PROJ
echo UPATH = $UPATH

#echo "Delete calibrated data if it already exists..."
rm -rf $UPATH/calibrated/

echo "Running ScriptForPI.py..."
cd $UPATH/script/
casa --pipeline -c *scriptForPI.py
sleep 5

echo "Calibrated data should appear here:"
echo "ls $UPATH/calibrated/"
ls $UPATH/calibrated/

# Record the end time and print execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Script execution time: $duration seconds"

echo
echo "Bash script complete."
echo