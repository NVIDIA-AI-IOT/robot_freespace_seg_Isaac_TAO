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

import yaml
import os


class YamlReader():
    def __init__(self, yaml_file_path):
        self.load_yaml_file(yaml_file_path)
        
    def load_yaml_file(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            self.yaml_file = yaml.safe_load(file)
    
    def get_floor_texture_folder(self):
        floor_texture_folder = self.yaml_file['floor_texture_folder']['path']
        return floor_texture_folder
    
    def get_wall_texture_folder(self):
        wall_texture_folder = self.yaml_file['wall_texture_folder']['path']
        return wall_texture_folder
    
    def get_wall_group_list(self):
        return self.yaml_file['wall_group']['wall_prims']

    def get_randomization_object_list(self):
        return self.yaml_file['randomization_objects']['warehouse_objects']

    def get_additional_objects_list(self):
        return self.yaml_file['additional_objects']['ycb_objects']

    def get_floor_group_list(self):
        return self.yaml_file['floor_group']['floor_prims']

    def get_light_group_list(self):
        return self.yaml_file['light_group']['light_prims']

    def get_randomization_objects(self):
        return self.yaml_file['randomization_objects']['existing_objects']

 
 
