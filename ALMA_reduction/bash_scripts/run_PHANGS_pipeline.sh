#!/usr/bin/env bash

export NAME=$1
export OBJID=$2

# set up output logging file
output_file="/arc/projects/salvage/ALMA_reduction/logs/run_PHANGS_pipeline_${OBJID}_$(date +'%Y%m%d_%H%M%S').txt"
exec > "$output_file" 2>&1

echo
echo "Execute bash script."
echo
start_time=$(date +%s)

# conda activate almaredux

# test/report environment variables
echo NAME = $NAME
echo OBJID = $OBJID

echo "Running CASA pipeline..."
cd /arc/projects/salvage/phangs_imaging_scripts/
casa -c run_casa_pipeline_phangs-alma_salvage.py
sleep 5

# Record the end time and print execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
#duration_minutes=$(echo "scale=2; $duration / 60" | bc)
echo "Script execution time: $duration seconds"

echo
echo "Bash script complete."
echo