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
#all copies or substantial portions of the Software.

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

def train_validation_splitter(train_images_path, train_labels_path, validation_images_path, validation_labels_path, split_ratio=0.85):
    
    train_images_list = os.listdir(train_images_path)
    train_label_list = os.listdir(train_labels_path)

    train_validation_split = split_ratio

    num_validation = int((1-train_validation_split)*len(train_images_list))
    files_to_move = random.sample(range(len(train_images_list)), num_validation)

    for file_index in files_to_move:
        train_image = train_images_list[file_index]
        train_label = train_label_list[file_index]

        source_image = os.path.join(train_images_path, train_image)
        source_label = os.path.join(train_labels_path, train_label)

        shutil.move(source_image, validation_images_path)
        shutil.move(source_label, validation_labels_path)


    validation_images_list = os.listdir(validation_images_path)
    validation_labels_list = os.listdir(validation_labels_path)

    print(validation_images_list == validation_labels_list)
    print(len(validation_images_list))
    print("Done!!")

if __name__ == "__main__":
    "Typical usage"
    parser = argparse.ArgumentParser("Split Data to Train and Validation Set")
    parser.add_argument("--train_images", type=str, help="Train Images folder for TAO")
    parser.add_argument("--train_labels", type=str, help="Train Labels folder for TAO")
    parser.add_argument("--val_images", type=str, help="Validation Images folder for TAO")
    parser.add_argument("--val_labels", type=str, help="Validation Labels folder for TAO")
    
    args, unknown_args = parser.parse_known_args()

    train_validation_splitter(train_images_path=args.train_images,
    			train_labels_path=args.train_labels, 
    			validation_images_path=args.val_images,
                validation_labels_path=args.val_labels)
