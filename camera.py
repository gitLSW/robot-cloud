import math
import numpy as np
# from omni.isaac.core.utils.prims import create_prim, define_prim, get_prim_at_path
# from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.numpy.rotations as rot_utils
# from omni.isaac.sensor import Camera
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.synthetic_utils")
from omni.isaac.synthetic_utils import SyntheticDataHelper
sd_helper = SyntheticDataHelper()


class VideoFeed:
    
    def __init__(self, position, target, width: int, height: int, freq: int, headless: bool = True, camera_path: str = "/World/Camera"):
        """
        Args:
            id: The id of the camera
            width: The horizontal image resolution in pixels
            height: The vertical image resolution in pixels
            fov: The field of view of the camera
            near: The near plane distance
            far: The far plane distance
        """
        
        self._width = width
        self._height = height
        self.camera_prim_path = camera_path
        
        set_camera_view(eye=position, target=target, camera_prim_path="/OmniverseKit_Persp")
        self.viewport = get_active_viewport()

        # dir_vec = np.array(target) - np.array(position)
        # dir_vec /= np.linalg.norm(dir_vec) # Normalize to Unit Vec
        # x_axis = np.array([1, 0, 0])
        # y_axis = np.array([0, 1, 0])
        # z_axis = np.array([0, 0, 1])
        # x_rotation_matrix = np.cross(x_axis, dir_vec)
        # y_rotation_matrix = np.cross(y_axis, dir_vec)
        # z_rotation_matrix = np.cross(z_axis, dir_vec)

        # self.camera_prim = Camera(
        #     prim_path=camera_path,
        #     position=position,
        #     # frequency=freq,
        #     resolution=(width, height),
        #     orientation=rot_utils.rot_matrices_to_quats(rotation_matrices=np.array([x_rotation_matrix, y_rotation_matrix, z_rotation_matrix])),
        # )

        # # Set as current camera
        # if headless:
        #     viewport_interface = omni.kit.viewport_legacy.get_viewport_interface()
        #     self.viewport = viewport_interface.get_viewport_window()
        # else:
        #     viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
        #     list_viewports = omni.kit.viewport_legacy.get_viewport_interface().get_instance_list()
        #     new_viewport_name = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window_name(
        #         viewport_handle
        #     )
        #     self.viewport = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle) 
        #     window_width = 200
        #     window_height = 200
        #     self.viewport.set_window_size(window_width, window_height)
        #     self.viewport.set_window_pos(800 , window_height*(len(list_viewports)-2))

        # self.viewport.set_active_camera(self.camera_prim_path)
        # self.viewport.set_texture_resolution(self._width, self._height)

    def get_image(self):
        # Get ground truths
        gt = sd_helper.get_groundtruth(
            [
                "rgb",
                #"depthLinear",
                "depth",
                #"boundingBox2DTight",
                #"boundingBox2DLoose",
                "instanceSegmentation",
                #"semanticSegmentation",
                #"boundingBox3D",
                #"camera",
                #"pose"
            ],
            self.viewport,
        )

        # print("Camera params", sd_helper.get_camera_params(self.viewport))

        segmentation_mask = gt["instanceSegmentation"]
        rgb = gt["rgb"]
        depth = gt["depth"]
        return rgb, depth, segmentation_mask
    
    def get_pose(self):
        transform_matrix = sd_helper.get_camera_params(self.viewport)["pose"]
        return transform_matrix

    def set_prim_pose(self, position, orientation):
        properties = self.camera_prim.GetPropertyNames()
        if "xformOp:translate" in properties:
            translate_attr = self.camera_prim.GetAttribute("xformOp:translate")
            translate_attr.Set(Gf.Vec3d(position))
        if "xformOp:orient" in properties:
            orientation_attr = self.camera_prim.GetAttribute("xformOp:orient")
            orientation_attr.Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))