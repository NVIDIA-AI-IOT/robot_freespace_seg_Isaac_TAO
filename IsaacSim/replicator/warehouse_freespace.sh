# This is the path where you cloned this repo, specify the Isaac Sim folder path
SCRIPT_PATH="${PWD}/warehouse_freespace.py"
YAML_PATH="${PWD}/warehouse_freespace.yaml"

# This is the path where the output will be saved for simple room
OUTPUT_FOLDER="${PWD}/warehouse_freespace"

cd ../../..

./python.sh $SCRIPT_PATH  --num_frames 10 --yaml_path $YAML_PATH --data_dir $OUTPUT_FOLDER

./python.sh $SCRIPT_PATH --num_frames 10 --yaml_path $YAML_PATH --light_dr True --data_dir $OUTPUT_FOLDER

./python.sh $SCRIPT_PATH --num_frames 10 --yaml_path $YAML_PATH --wall_dr color --data_dir $OUTPUT_FOLDER

./python.sh $SCRIPT_PATH --num_frames 10 --yaml_path $YAML_PATH --floor_dr color --data_dir $OUTPUT_FOLDER

./python.sh $SCRIPT_PATH --num_frames 10 --yaml_path $YAML_PATH --wall_dr color --floor_dr color --data_dir $OUTPUT_FOLDER