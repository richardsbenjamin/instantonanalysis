#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/instantonanalysis"
OUTPUT_DIR="${MODULE_DIR}/outputs"
OUTPUT_FILE="${OUTPUT_DIR}/calculations.out"


#############################################################
cd ${MODULE_DIR}
export PYTHONPATH=${HOME_DIR}

# Delete the output file if it already exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

RUN_CMD="python scripts/calculations.py"
${RUN_CMD}

# # If output file is given, redirect output
# if [[ -n "${OUTPUT_FILE}" ]]; then
#     ${RUN_CMD} &>> ${OUTPUT_FILE}
# else
#     ${RUN_CMD}
# fi