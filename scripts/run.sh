#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No run name provided."
    echo "Usage: ./run.sh <run_name>"
    exit 1
fi

RUN="$1"
VERBOSE_MODE=false

if [[ "$2" == "-v" || "$2" == "-verbose" ]]; then
    VERBOSE_MODE=true
fi


# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/instantonanalysis"
OUTPUT_DIR="${MODULE_DIR}/outputs"

if [ "$VERBOSE_MODE" = true ]; then
    OUTPUT_FILE=""
    echo "Running in Verbose Mode (printing to console)..."
else
    OUTPUT_FILE="${OUTPUT_DIR}/${RUN}.out"
fi

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/${RUN}.py"
${RUN_CMD}

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
    echo "Done. Output saved to ${OUTPUT_FILE}"
else
    ${RUN_CMD}
fi