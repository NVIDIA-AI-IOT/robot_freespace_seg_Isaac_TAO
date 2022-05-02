# freespace_seg


For generating simple room data, create an executable for the `simple_room_freespace.sh` by running `chmod +x simple_room_freespace.sh`

Edit the script to ensure correcpt paths are set for Isaac Sim Installation directory and the Outpur Folder.

Follow the same steps for `warehouse_freespace.sh`. 

Randomizations and number of images can be controlled in the SDG script via the command line parameters.

The YAML file corresponding to each scenario can be used to add more objects to the scene by specifying their nucleus server path.

After generating the data, scripts from `utils` can be used to postprocess the data and prpeare for training with `TAO`


