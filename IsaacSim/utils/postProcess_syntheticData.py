# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

#  SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

 #Permission is hereby granted, free of charge, to any person obtaining a
 #copy of this software and associated documentation files (the "Software"),
 #to deal in the Software without restriction, including without limitation
 #the rights to use, copy, modify, merge, publish, distribute, sublicense,
 #and/or sell copies of the Software, and to permit persons to whom the
 #Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#DEALINGS IN THE SOFTWARE.

import os
import cv2
import argparse

def rename_files(parent_folder_path, prefix):
    if(prefix == 'simple_room' or prefix == 'warehouse'):
        rgb_folder_path = os.path.join(parent_folder_path, 'rgb')
        for filename in os.listdir(rgb_folder_path):
            if(filename.startswith(prefix)):
                break
            else:
                print("Renaming: ", filename)
                newName = prefix + filename
                os.rename(os.path.join(rgb_folder_path,filename),os.path.join(rgb_folder_path, newName))
        print("Renamed RGB files....")

        semantic_folder_path = os.path.join(parent_folder_path, 'semantic')
        for filename in os.listdir(semantic_folder_path):
            if(filename.startswith(prefix)):
                break
            else:
                print("Renaming: ", filename)
                newName = prefix + filename
                os.rename(os.path.join(semantic_folder_path,filename),os.path.join(semantic_folder_path, newName))
        print("Renamed Semantic files....")

    else:
        print("Please select a scenario from simple_room or warehouse...")

def makeDirectory(parent_folder_path):
    label_folder_path = os.path.join(parent_folder_path, 'labels')
    try:
        os.mkdir(label_folder_path)
        print("Created the label folder, now converting to masks for TAO format....")
    except OSError as error:
        print("The label folder already exists, converting the masks to TAO format....")


def create_simpleRoom_labels(folder_path):
    semantic_location = os.path.join(folder_path, 'semantic')
    label_location = os.path.join(folder_path, 'labels')
    array_list = os.listdir(semantic_location)

    for array in array_list:
        arr_path = os.path.join(semantic_location, array)
        print(arr_path)

        arr_load = cv2.imread(arr_path)
        arr_load = arr_load[:,:,0]

        # Set the floor to an ID of 1, all remaining objects in the scene are set to zero
        arr_load[arr_load != 25] = 0
        arr_load[arr_load == 25] = 1
        
        image_name = array[:-4] + '.png'
        image_path = os.path.join(label_location, image_name)
        cv2.imwrite(image_path, arr_load)
    print("Generated TAO Masks for Simple Room....")


def create_warehouse_labels(folder_path):
    semantic_location = os.path.join(folder_path, 'semantic')
    label_location = os.path.join(folder_path, 'labels')
    array_list = os.listdir(semantic_location)

    for array in array_list:
        arr_path = os.path.join(semantic_location, array)
        print(arr_path)
        arr_load = cv2.imread(arr_path)
        arr_load = arr_load[:,:,2]

        # Set the floor to an ID of 1, all remaining objects in the scene are set to zero
        arr_load[arr_load != 169] = 0
        arr_load[arr_load == 169] = 1
        
        image_name = array[:-4] + '.png'
        image_path = os.path.join(label_location, image_name)
        cv2.imwrite(image_path, arr_load)
    print("Generated TAO Masks for Warehouse....")    

def createTao_labels(parent_folder_path, prefix):
    if(prefix == 'simple_room'):
        create_simpleRoom_labels(parent_folder_path)
    elif(prefix == 'warehouse'):
        create_warehouse_labels(parent_folder_path)

def create_labels(parent_folder_path, prefix):
    makeDirectory(parent_folder_path)
    createTao_labels(parent_folder_path, prefix)


#folder_path = "/home/karma/synthetic_data/script_data/automated/simple_room"
#rename_files(parent_folder_path=folder_path, prefix='simpleRoom')

if __name__ == "__main__":
    "Typical usage"
    parser = argparse.ArgumentParser("Dataset Postprocessing")
    parser.add_argument("--scenario", type=str, help="select from simple_room or warehouse")
    parser.add_argument("--viewport_directory", type=str, help="Path to Viewport which contains rgb and semantic labels sub folders")
    
    args, unknown_args = parser.parse_known_args()

    rename_files(parent_folder_path=args.viewport_directory, prefix=args.scenario)
    create_labels(parent_folder_path=args.viewport_directory, prefix=args.scenario)