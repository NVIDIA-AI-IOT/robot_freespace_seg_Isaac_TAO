#!/bin/bash

# This is the path where Isaac Sim is installed which contains the python.sh script
ISAAC_SIM_PATH='/home/karma/isaac_replicator/omni_isaac_sim/_build/linux-x86_64/release'

# This is the path where you cloned this repo, specify the Isaac Sim folder path
SCRIPT_PATH="${PWD}/IsaacSim/sdg/warehouse_freespace.py"
YAML_PATH="${PWD}/IsaacSim/sdg/warehouse_freespace.yaml"

# This is the path where the output will be saved for simple room
OUTPUT_FOLDER="${PWD}/IsaacSim/warehouse_freespace"

echo "Generating Data For Warehouse"  

cd $ISAAC_SIM_PATH

echo $PWD

./python.sh $SCRIPT_PATH --scenario omniverse://localhost/Isaac/Environments/Simple_Warehouse/full_warehouse.usd --num_frames 1000 --max_queue_size 500 --floor_dr texture --wall_dr texture --data_dir $OUTPUT_FOLDER --yaml_path $YAML_PATH

./python.sh $SCRIPT_PATH --scenario omniverse://localhost/Isaac/Environments/Simple_Warehouse/full_warehouse.usd  --num_frames 1000 --max_queue_size 500 --floor_dr color --wall_dr texture --data_dir $OUTPUT_FOLDER --yaml_path $YAML_PATH

./python.sh $SCRIPT_PATH --scenario omniverse://localhost/Isaac/Environments/Simple_Warehouse/full_warehouse.usd  --num_frames 1000 --max_queue_size 500 --floor_dr texture --wall_dr color --data_dir $OUTPUT_FOLDER --yaml_path $YAML_PATH

./python.sh $SCRIPT_PATH --scenario omniverse://localhost/Isaac/Environments/Simple_Warehouse/full_warehouse.usd  --num_frames 1000 --max_queue_size 500 --floor_dr color --wall_dr color --data_dir $OUTPUT_FOLDER --yaml_path $YAML_PATH

./python.sh $SCRIPT_PATH --scenario omniverse://localhost/Isaac/Environments/Simple_Warehouse/full_warehouse.usd  --num_frames 1000 --max_queue_size 500 --data_dir $OUTPUT_FOLDER --yaml_path $YAML_PATH


