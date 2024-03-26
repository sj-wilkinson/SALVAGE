#!/usr/bin/env bash

source ~/.bashrc

export ID=$1
export PROJ=$2
export UPATH=$3

is_recent_file() {
    local file="$1"
    local file_time=$(stat -c %Y "$file")
    local current_time=$(date +%s)
    local time_diff=$((current_time - file_time))
    local threshold=$((4 * 60 * 60))  # 4 hours in seconds

    if [ "$time_diff" -le "$threshold" ]; then
        return 0  # File is recent
    else
        return 1  # File is not recent
    fi
}

delete_recent_files() {
    local dir="$1"
    
    # Iterate through all files and directories in the given directory
    for entry in "$dir"/*; do
        if [ -f "$entry" ]; then  # If it's a file
            if is_recent_file "$entry"; then
                echo "Deleting recent file: $entry"
                rm "$entry"
            fi
        elif [ -d "$entry" ]; then  # If it's a directory
            delete_recent_files "$entry"  # Recursively call the function for subdirectories
        fi
    done
}

# set up output logging file
if [ ! -d "/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/" ]; then mkdir "/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/"; fi
output_file="/arc/projects/salvage/ALMA_reduction/logs/$(date +'%Y%m%d')/run_restore_calibration_${ID}_$(date +'%Y%m%d_%H%M%S').txt"
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

# Since headless jobs do not have priority on the science platform
# Delete all files in UPATH that have been altered in the last 4 hours
# This should avoid scriptForPI.py building off something that is only half done
delete_recent_files "$UPATH"

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