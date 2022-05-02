# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
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
import random
import numpy as np
import shutil
import argparse


def copy_training_data(source_folder_path, tao_train_images_path, tao_train_labels_path):
    "Here the source folder contains the rgb and labels directory which is generated after postprocessing"
    image_folder = os.path.join(source_folder_path, 'Viewport/rgb')
    image_list = os.listdir(image_folder)

    label_folder = os.path.join(source_folder_path, 'Viewport/labels')
    label_list = os.listdir(label_folder)

    num_images = len(os.listdir(image_folder))
    num_labels = len(os.listdir(label_folder))

    for num in range(0,num_images):
        current_image = image_list[num]
        current_label = label_list[num]
        #print(num)
        image_source = os.path.join(image_folder, current_image)
        label_source = os.path.join(label_folder, current_label)

        shutil.copy(image_source, tao_train_images_path)
        shutil.copy(label_source, tao_train_labels_path)

    print(os.listdir(tao_train_images_path)== os.listdir(tao_train_labels_path))


if __name__ == "__main__":
    "Typical usage"
    parser = argparse.ArgumentParser("Move Data")
    parser.add_argument("--sim_source_folder", type=str, help="Path to folder which contains Viewport and postprocessed data")
    parser.add_argument("--tao_train_images_folder", type=str, help="Destination folder for training images with TAO")
    parser.add_argument("--tao_train_labels_folder", type=str, help="Destination folder for training labels with TAO")
    
    args, unknown_args = parser.parse_known_args()

    copy_training_data(source_folder_path=args.sim_source_folder,
    			tao_train_images_path=args.tao_train_images_folder, 
    			tao_train_labels_path=args.tao_train_labels_folder)
    			

