import asyncio
import copy
import os
import torch
import signal
import argparse

import carb
import omni
from omni.isaac.kit import SimulationApp

# Default Rendering Parameters
CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1280, "height": 720}

kit = SimulationApp(launch_config=CONFIG)

from omni.isaac.synthetic_utils import SyntheticDataHelper, NumpyWriter
import omni.isaac.dr as dr
from omni.isaac.core.utils.nucleus import find_nucleus_server
from pxr import Gf, UsdGeom
import numpy as np 

from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import random
from random import randint
from pxr import Gf, Sdf, UsdPhysics
from pxr import PhysxSchema, PhysicsSchemaTools
from omni.physx.scripts import utils
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.usd.commands import DeletePrimsCommand
from omni.isaac.core.utils.nucleus import _list
from yaml_reader import YamlReader


class RandomScenario(torch.utils.data.IterableDataset):

    def __init__(self, scenario_path, data_dir, max_queue_size, floor_dr, wall_dr, yaml_path):
        self.vpi = omni.kit.viewport.get_viewport_interface()
        self.sd_helper = SyntheticDataHelper()
        self.dr = dr
        self.writer_helper = NumpyWriter
        self.dr.commands.ToggleManualModeCommand().do()
        self.stage = kit.context.get_stage()
        self.result = True

        if scenario_path is None:
            self.result, nucleus_server = find_nucleus_server()
            if self.result is False:
                carb.log_error("Could not find nucleus server with /Isaac Folder")
                return
            self.asset_path = nucleus_server + "/Users"
            scenario_path = self.asset_path + "/no_carter_warehouse_no_objects.usd"

        self.yaml_reader = YamlReader(yaml_path)
        self.scenario_path = scenario_path
        self.max_queue_size = max_queue_size
        self.data_writer = None
        self.data_dir = data_dir
        self.get_num_images_output()

        self.wall_dr = wall_dr
        self.floor_dr = floor_dr
        
        self._setup_world(scenario_path)
        self.cur_idx = 0
        self.exiting = False
        self._sensor_settings = {}

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, *args, **kwargs):
        print("Exiting Dataset Generation......")
        self.exiting = True

    async def load_stage(self, path):
        await omni.usd.get_context().open_stage_async(path)

    def _setup_world(self, scenario_path):
        # Load Scenario
        setup_task = asyncio.ensure_future(self.load_stage(scenario_path))
        while not setup_task.done():
            kit.update()

        #self.group_objects_in_scene()
        self.set_carter_camera_viewport()
        self.add_objects_in_scene()
        self.create_dr_components()

        kit.update()

    def get_num_images_output(self):
        # Check the number of existing images in the output directory
        rgb_image_folder = os.path.join(self.data_dir, 'Viewport/rgb')
        try:
            self.num_images = len(os.listdir(rgb_image_folder))
        except:
            print("No images existing in output directory.....")
            self.num_images = 0
        print("Number of images pre-existing in the Output Directory: {}".format(self.num_images)) 
    

    def set_carter_camera_viewport(self):
        self.vpi = omni.kit.viewport.get_viewport_interface()
        left_camera_path = '/World/Carter_ROS/chassis_link/camera_mount/carter_camera_stereo_left'
        self.vpi.get_viewport_window().set_active_camera(left_camera_path)
        self.viewport = omni.kit.viewport.get_default_viewport_window()


    def group_objects_in_scene(self):
        self.group_wall_prims()
        self.group_floor_prims()
        self.group_light_prims()
        #self.add_group_objects()

    def group_wall_prims(self):
        self.wall_group = UsdGeom.Xformable(create_prim("/Root/Group_Wall", "Xform"))
        wall_group = self.yaml_reader.get_wall_group_list()
        
        path_to_dest = "/Root/Group_Wall"
        for source in wall_group:
            move_dict = {}
            move_dict[source] = path_to_dest + source[6:] 
            omni.kit.commands.execute("MovePrimsCommand", paths_to_move=move_dict)
        print("Successfully added prims to the same parent for the wall...........")


    def group_floor_prims(self):
        self.floor_group = UsdGeom.Xformable(create_prim("/Root/Group_Floor", "Xform"))
        floor_group = self.yaml_reader.get_floor_group_list()
    
        path_to_dest = "/Root/Group_Floor"
        
        for source in floor_group:
            move_dict = {}
            move_dict[source] = path_to_dest + source[6:] 
            omni.kit.commands.execute("MovePrimsCommand", paths_to_move=move_dict)
        print("Successfully added prims to the same parent for the floor...........")


    def group_light_prims(self):
        self.light_group = UsdGeom.Xformable(create_prim("/Root/Group_Light", "Xform"))
        light_group = self.yaml_reader.get_light_group_list()

        path_to_dest = "/Root/Group_Light"
        for source in light_group:
            move_dict = {}
            move_dict[source] = path_to_dest + source[6:] 
            omni.kit.commands.execute("MovePrimsCommand", paths_to_move=move_dict)
        print("Successfully grouped the lights together............")


    def load_single_asset(self, object_transform_path, usd_object_path):
        translate_x = 150 * random.random()
        translate_y = 150 * random.random()
        asset = None
        try:
            _asset = create_prim(object_transform_path, "Xform",
                                position=np.array([150 + translate_x, 175 + translate_y, 0]), 
                                orientation=euler_angles_to_quat(np.array([0, -0.0, 0]), degrees=True),
                                usd_path=usd_object_path)
            asset = _asset
            print("Adding {} object to the scene".format(usd_object_path))
            self.asset_list.append(asset)
            #utils.setRigidBody(_asset, "convexHull", False)
        except:
            print("NOT able to add {} object to the scene".format(usd_object_path))
            carb.log_warn("load_single_asset failure")

    def add_objects_in_scene(self):
        # Load and add objects into the scene
        objects_list = self.yaml_reader.get_randomization_object_list()
        self.asset_list = []
        indx = 0
        for num in range(0,2):
            for usd_object in objects_list:
                asset = None
                object_transform_path = f"/Root/object" + str(indx)
                self.load_single_asset(object_transform_path, usd_object)
                stage = kit.context.get_stage()
                object_prim = stage.GetPrimAtPath(object_transform_path)
                add_update_semantics(object_prim, "obstacle") 
                indx += 1

    def random_five_integer(self):
        # generate a random number for the DR seed
        range_begin = 10**4
        range_last = (10**5) - 1
        random_number = randint(range_begin, range_last)
        return random_number 

    def create_dr_components(self):
        # Create and set the Domain Randomization Components
        self.create_carter_randomization()
        self.create_objects_randomization()
        #self.create_floor_randomization()
        #self.create_wall_randomization()
        #self.create_light_randomization()
        #self.create_visbilty_randomization()
        kit.update()
        #self.save_stage()
        print("Successfully added object paths for Domain Randomization..............!")          

    def create_carter_randomization(self):
        carter_tranlsate_min_range, carter_translate_max_range = (-500, -200, 24), (500, 200, 24)
        carter_rotate_min_range, carter_rotate_max_range = (0, 0, 0), (0, 0 ,360)
        carter_path = '/World/Carter_ROS'
        self.camera_transform = self.dr.commands.CreateTransformComponentCommand(prim_paths=[carter_path], 
                                                                                translate_min_range=carter_tranlsate_min_range, 
                                                                                translate_max_range=carter_translate_max_range, 
                                                                                rotate_min_range=carter_rotate_min_range, 
                                                                                rotate_max_range=carter_rotate_max_range,
                                                                                duration=0.25, seed=self.random_five_integer()).do()

    async def get_floor_textures(self):
        texture_folder = self.yaml_reader.get_floor_texture_folder()
        self.floor_texture_files = await _list(texture_folder)       

    def create_floor_randomization(self):
        prim_path = "/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/Group_Floor"
        if self.floor_dr == 'texture':
            setup_task = asyncio.ensure_future(self.get_floor_textures())
            while not setup_task.done():
                kit.update() 
            print("Floor textures are: ", self.floor_texture_files)
            self.floor_texture_comp = self.dr.commands.CreateTextureComponentCommand(prim_paths=[prim_path],  
                                                                            texture_list=self.floor_texture_files[0],
                                                                            enable_project_uvw=False,
                                                                            duration=0.25,
                                                                            seed=self.random_five_integer()).do()
        elif self.floor_dr == 'color':                                                                    
            self.floor_color_comp = self.dr.commands.CreateColorComponentCommand(prim_paths=[prim_path],
                                                                        duration=0.25, 
                                                                       seed=self.random_five_integer()).do()
        else:
            print("No Domain Randomization for Floor.....")

    async def get_wall_textures(self):
        texture_folder = self.yaml_reader.get_wall_texture_folder()
        self.wall_texture_files = await _list(texture_folder) 

    def create_wall_randomization(self):
        prim_path = "/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/Group_LowerWall"
        if self.wall_dr == 'texture':
            setup_task = asyncio.ensure_future(self.get_wall_textures())
            while not setup_task.done():
                kit.update() 
            print("Wall textures are: ", self.wall_texture_files)
            self.wall_texture_comp = self.dr.commands.CreateTextureComponentCommand(prim_paths=[prim_path],  
                                                                            texture_list=self.wall_texture_files[0],
                                                                            enable_project_uvw=False,
                                                                            duration=0.25,
                                                                            seed=self.random_five_integer()).do()
            
            self.wall_visibility_comp = self.dr.commands.CreateVisibilityComponentCommand(prim_paths=[prim_path],
                                                                                duration=0.5).do()

        elif self.wall_dr == 'color':
            self.wall_color_comp = self.dr.commands.CreateColorComponentCommand(prim_paths=[prim_path],
                                                                        duration=0.25,
                                                                        seed=self.random_five_integer()).do()
        else:
            print("No Domain Randomization for Wall..........")


    def add_targets(self, dr_component):
        dr_comp_target = dr_component.GetPrimPathsRel()
        dr_comp_target.ClearTargets(True)
        kit.update()
        for asset in self.asset_list:
            dr_comp_target.AddTarget(asset.GetPrimPath())
        kit.update()

    def create_objects_randomization(self):
        self.objects_transform = self.dr.commands.CreateTransformComponentCommand(prim_paths=[], 
                                                                                polygon_points=[(-611, -1049, 0), (-611, 1183, 0),
                                                                                (680, 1280, 0), (680, 0, 0), (158, -16, 0),
                                                                                (158, -1000, 0)],
                                                                                translate_max_range=(0.0, 0.0, 0.0),
                                                                                rotate_min_range=[0,0,0], 
                                                                                rotate_max_range=[0, 0, 360],
                                                                                scale_min_range=[1.0, 1.0, 1.0],
                                                                                scale_max_range=[1.5, 1.5, 1.5],
                                                                                duration=1.0,
                                                                                seed=self.random_five_integer()).do()
        
        self.add_targets(self.objects_transform)  

    def create_light_randomization(self):
        light_list = ["/Root/RectLight"]
        self.light_comp = self.dr.commands.CreateLightComponentCommand(light_paths=light_list,
                                                                         duration=0.5,
                                                                         seed=self.random_five_integer()).do()
                                                                        

    def create_visbilty_randomization(self):
        self.visibility_comp = self.dr.commands.CreateVisibilityComponentCommand(prim_paths=[],
                                                                                num_visible_range=[5, 70], 
                                                                                duration=1.0).do()
        
        self.add_targets(self.visibility_comp)
        
    def save_stage(self):
        omni.usd.get_context().save_as_stage(os.getenv("HOME")+ '/Jetbot_room/warehouse_segmentation.usd', None)       
    
    def _capture_viewport(self, viewport_name, sensor_settings):
        print("capturing viewport:", viewport_name)
        viewport = self.vpi.get_viewport_window(self.vpi.get_instance(viewport_name))
        if not viewport:
            carb.log_error("Viewport Not found, cannot capture")
            return
        groundtruth = {
            "METADATA": {
                "image_id": str(self.cur_idx + self.num_images),
                "viewport_name": viewport_name,
                "DEPTH": {},
                "INSTANCE": {},
                "SEMANTIC": {},
                "BBOX2DTIGHT": {},
                "BBOX2DLOOSE": {},
            },
            "DATA": {},
        }

        gt_list = []
        if sensor_settings["rgb"]["enabled"]:
            gt_list.append("rgb")
        if sensor_settings["depth"]["enabled"]:
            gt_list.append("depthLinear")
        if sensor_settings["bbox_2d_tight"]["enabled"]:
            gt_list.append("boundingBox2DTight")
        if sensor_settings["bbox_2d_loose"]["enabled"]:
            gt_list.append("boundingBox2DLoose")
        if sensor_settings["instance"]["enabled"]:
            gt_list.append("instanceSegmentation")
        if sensor_settings["semantic"]["enabled"]:
            gt_list.append("semanticSegmentation")

        # on the first frame make sure sensors are initialized
        if self.cur_idx == 0:
            self.sd_helper.initialize(sensor_names=gt_list, viewport=viewport)
            kit.update()
            kit.update()
        # Render new frame
        kit.update()
        kit.update()
        kit.update()

        # Collect Groundtruth
        gt = self.sd_helper.get_groundtruth(gt_list, viewport, wait_for_sensor_data=0.25)

        user_semantic_label_map ={"floor":31, "forklift":32, "obstacle":33} 

        # RGB
        image = gt["rgb"]
        if sensor_settings["rgb"]["enabled"] and gt["state"]["rgb"]:
            groundtruth["DATA"]["RGB"] = gt["rgb"]

        # Depth
        if sensor_settings["depth"]["enabled"] and gt["state"]["depthLinear"]:
            groundtruth["DATA"]["DEPTH"] = gt["depthLinear"].squeeze()
            groundtruth["METADATA"]["DEPTH"]["COLORIZE"] = sensor_settings["depth"]["colorize"]
            groundtruth["METADATA"]["DEPTH"]["NPY"] = sensor_settings["depth"]["npy"]

        # Instance Segmentation
        if sensor_settings["instance"]["enabled"] and gt["state"]["instanceSegmentation"]:
            instance_data = gt["instanceSegmentation"][0]
            groundtruth["DATA"]["INSTANCE"] = instance_data
            groundtruth["METADATA"]["INSTANCE"]["WIDTH"] = instance_data.shape[1]
            groundtruth["METADATA"]["INSTANCE"]["HEIGHT"] = instance_data.shape[0]
            groundtruth["METADATA"]["INSTANCE"]["COLORIZE"] = sensor_settings["instance"]["colorize"]
            groundtruth["METADATA"]["INSTANCE"]["NPY"] = sensor_settings["instance"]["npy"]

        # Semantic Segmentation
        if sensor_settings["semantic"]["enabled"] and gt["state"]["semanticSegmentation"]:
            semantic_data = gt["semanticSegmentation"]
            semantic_data[semantic_data == 65535] = 0  # deals with invalid semantic id
            mapped_data = self.sd_helper.get_mapped_semantic_data(gt["semanticSegmentation"], user_semantic_label_map)
            mapped_data = np.array(mapped_data)
            print(np.unique(mapped_data, return_counts=True))
            mapped_data[mapped_data < 31] = 0
            mapped_data[mapped_data == 31] = 1
            mapped_data[mapped_data == 32] = 15
            mapped_data[mapped_data == 33] = 10
            groundtruth["DATA"]["SEMANTIC"] = mapped_data
            semantic_ids = self.sd_helper.get_semantic_ids(gt["semanticSegmentation"])
            semantic_id_map = self.sd_helper.get_semantic_label_map(semantic_ids)
            print(semantic_id_map)
            groundtruth["METADATA"]["SEMANTIC"]["WIDTH"] = semantic_data.shape[1]
            groundtruth["METADATA"]["SEMANTIC"]["HEIGHT"] = semantic_data.shape[0]
            groundtruth["METADATA"]["SEMANTIC"]["COLORIZE"] = sensor_settings["semantic"]["colorize"]
            groundtruth["METADATA"]["SEMANTIC"]["NPY"] = sensor_settings["semantic"]["npy"]

        # 2D Tight BBox
        if sensor_settings["bbox_2d_tight"]["enabled"] and gt["state"]["boundingBox2DTight"]:
            groundtruth["DATA"]["BBOX2DTIGHT"] = gt["boundingBox2DTight"]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["COLORIZE"] = sensor_settings["bbox_2d_tight"]["colorize"]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["NPY"] = sensor_settings["bbox_2d_tight"]["npy"]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["WIDTH"] = image.shape[1]
            groundtruth["METADATA"]["BBOX2DTIGHT"]["HEIGHT"] = image.shape[0]

        # 2D Loose BBox
        if sensor_settings["bbox_2d_loose"]["enabled"] and gt["state"]["boundingBox2DLoose"]:
            groundtruth["DATA"]["BBOX2DLOOSE"] = gt["boundingBox2DLoose"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["COLORIZE"] = sensor_settings["bbox_2d_loose"]["colorize"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["NPY"] = sensor_settings["bbox_2d_loose"]["npy"]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["WIDTH"] = image.shape[1]
            groundtruth["METADATA"]["BBOX2DLOOSE"]["HEIGHT"] = image.shape[0]

        self.data_writer.q.put(groundtruth)
        return image

    def __iter__(self):
        return self

    def __next__(self):

        # Enable/disable sensor output and their format
        sensor_settings_viewport = {
            "rgb": {"enabled": True},
            "depth": {"enabled": False, "colorize": True, "npy": True},
            "instance": {"enabled": False, "colorize": True, "npy": True},
            "semantic": {"enabled": True, "colorize": True, "npy": False},
            "bbox_2d_tight": {"enabled": False, "colorize": True, "npy": True},
            "bbox_2d_loose": {"enabled": False, "colorize": True, "npy": True},
        }
        self._sensor_settings["Viewport"] = copy.deepcopy(sensor_settings_viewport)

        # step once and then wait for materials to load
        self.dr.commands.RandomizeOnceCommand().do()
        kit.update()

        from omni.isaac.core.utils.stage import is_stage_loading

        while is_stage_loading():
            kit.update()

        num_worker_threads = 4

        # Write to disk
        if self.data_writer is None:
            print(f"Writing data to {self.data_dir}")
            self.data_writer = self.writer_helper(self.data_dir, num_worker_threads, 
                                                self.max_queue_size, self._sensor_settings)
            self.data_writer.start_threads()

        image = self._capture_viewport("Viewport", self._sensor_settings["Viewport"])

        self.cur_idx += 1
        return image


if __name__ == "__main__":
    "Typical usage"
    parser = argparse.ArgumentParser("Dataset generator")
    parser.add_argument("--scenario", type=str, help="Scenario to load from omniverse nucleus server")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to record")
    parser.add_argument(
        "--data_dir", type=str, default=os.getcwd() + "/simple_room_freespace", help="Location where data will be output"
    )
    parser.add_argument("--max_queue_size", type=int, default=500, help="Max size of queue to store and process data")
    parser.add_argument("--floor_dr", type=str, default=None, help="Defalut is no randomization, options are color/texture")
    parser.add_argument("--wall_dr", type=str, default=None, help="Defalut is no randomization, options are color/texture")
    parser.add_argument("--yaml_path", type=str, help="YAML file containing the scene setup details")
    
    args, unknown_args = parser.parse_known_args()

    dataset = RandomScenario(scenario_path=args.scenario, data_dir=args.data_dir, 
        max_queue_size=args.max_queue_size, floor_dr=args.floor_dr, wall_dr=args.wall_dr, yaml_path = args.yaml_path)

    if dataset.result:
        # Iterate through dataset and visualize the output
        print("Loading materials. Will generate data soon...")
        for image in dataset:
            print("ID: ", dataset.cur_idx)
            if dataset.cur_idx == args.num_frames:
                break
            if dataset.exiting:
                break

        # wait until done
        dataset.data_writer.stop_threads()
    # cleanup
    kit.close()



