#!/usr/bin/env bash

export NAME=$1
export OBJID=$2
export MSFILE=$3

# set up output logging file
if [ ! -d "/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/" ]; then mkdir "/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/"; fi
output_file="/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/run_PHANGS_pipeline_${OBJID}_$(date +'%Y%m%d_%H%M%S').txt"
exec > "$output_file" 2>&1

echo
echo "Execute bash script."
echo
start_time=$(date +%s)

# test/report environment variables
echo NAME = $NAME
echo OBJID = $OBJID
echo MSFILE = $MSFILE

echo "Splitting out data manually..."
cd /arc/projects/salvage/ALMA_reduction/py_scripts/
casa -c /arc/projects/salvage/ALMA_reduction/py_scripts/run_manual_split-salvage.py
sleep 5

echo "Running CASA pipeline..."
cd /arc/projects/salvage/phangs_imaging_scripts/
casa -c run_casa_pipeline_phangs-alma_salvage.py
sleep 5

echo "Deleting existing any ms.split output file."
echo output_vis = $output_vis
echo "rm -rf $output_vis"
rm -rf $output_vis
sleep 2

# Record the end time and print execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Script execution time: $duration seconds"

# make completion file
touch "/arc/projects/salvage/ALMA_reduction/salvage_completion_files/${OBJID}_imaging_complete.txt"

echo
echo "Bash script complete."
echo