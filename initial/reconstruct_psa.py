# %%

from unittest import result
import numpy as np
import pyaoi
import cv2
import os
import time

from pyaoi.calibration import CameraCalibration,CalibrationData
from pyaoi.calibration import ProjectorCalibration
from pyaoi.reconstruction import Geometry
from pyaoi.hardware import projector
from pyaoi.hardware.projector import MirrorType

# from pyaoi.utils import save_pointcloud,load_config

import matplotlib.pyplot as plt

from constants import *
import pptk

import open3d as o3d


def get_zrange():
	selected = v.get("selected")
	if len(selected) == 0:
		print("no selection")
		return
	# print(selected)
	selected_points = point_cloud[selected,2]
	print(f"min_distance, max_distance = {round(selected_points.min())},{round(selected_points.max())}")

def show_selection():
    mask_pattern = cv2.cvtColor(white_pattern,cv2.COLOR_GRAY2BGR)
    selected = v.get("selected")
    selected_location = (point_locations[0][selected],point_locations[1][selected])
    mask_pattern[selected_location[0],selected_location[1]] = (255,0,0)
    pyaoi.imshow([mask_pattern])	
    stamp = time.strftime("%Y%m%d%H%M%S")[2:]
    pyaoi.imwrite(mask_pattern,f"{fsa_result_path}/selection_{stamp}.bmp")

def remove_select():
	global point_cloud,point_color
	selected = v.get("selected")
	if len(selected) == 0:
		print("no selection")
		return

if __name__ == "__main__":

    target_index = 13

    white_pattern = pyaoi.imread(fsa_data_path + "/hor/0.png")

   

    hor_phase_path = psa_result_path + "\\phase_h.tiff"
    hor_mask_path = psa_result_path + "\\mask_h.bmp"

    ver_phase_path = psa_result_path + "\\phase_v.tiff"
    ver_mask_path = psa_result_path + "\\mask_v.bmp"



    oversample = 4

    #加载投影仪，相机标定文件
    cam_calib_data = CalibrationData(cam_data_path)
    proj_calib_data = CalibrationData(proj_data_path)


    min_distance,max_distance = None,None
    # min_distance, max_distance = 658273,684771


    geometry = Geometry(min_distance=min_distance,max_distance=max_distance)
    # 添加主视角,生成点云图在该坐标系体系下
    tstart = time.time()
    geometry.set_origin_view(proj_calib_data,generate_ray=True,
                            viewtype=MirrorType.ORTHOGONAL,
                            oversample_rows=oversample,
                            oversample_columns=oversample)   

    print("set origin view time:",time.time()-tstart)

    #添加相机视角
    tstart = time.time()
    view_id = geometry.add_view(cam_calib_data,generate_ray=True)
    print("add view time:",time.time()-tstart)  

    # get phase
    phase_hor = pyaoi.imread(hor_phase_path)
    phase_ver = pyaoi.imread(ver_phase_path)
    disparity_hor = phase_hor*proj_calib_data.model_resolution[0]
    disparity_ver = phase_ver*proj_calib_data.model_resolution[1]


    # get mask of phase
    phase_mask_hor = pyaoi.imread(hor_mask_path)
    phase_mask_ver = pyaoi.imread(ver_mask_path)
    phase_mask = phase_mask_hor
    phase_mask[phase_mask_ver != 255] = 0


    # disparity oversample, mask invalid points
    disparity_hor = np.round(disparity_hor*oversample).astype("int")
    disparity_ver = np.round(disparity_ver*oversample).astype("int")


    disparity_hor[phase_mask != 255] = -1
    disparity_ver[phase_mask != 255] = -1




    #生成点云
    tstart = time.time()
    point_cloud,point_color,distance_map,point_locations = geometry.generate_pointcloud(view_id,disparity_ver,disparity_hor,white_pattern=white_pattern)
    print("generate cloud time:",time.time()-tstart)
    print(point_cloud.shape)


    v = pptk.viewer(point_cloud)
    v.attributes(point_color/255,point_cloud[:,2])

    # invalid_mask = np.full_like(distance_map,True,dtype=np.bool8)
    # invalid_mask[point_locations] = False
    # pyaoi.imshow(invalid_mask*255)
    # pyaoi.imshow(pyaoi.convert_float_image(distance_map,invalid_mask=invalid_mask, colormap=cv2.COLORMAP_HSV))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(point_color/255)
    o3d.io.write_point_cloud(psa_result_path+"/pointcloud_psa.ply",pcd)
    pyaoi.imwrite(distance_map,psa_result_path+"/distancemap_psa.tiff")
    pyaoi.imwrite(pyaoi.convert_float_image(distance_map,invalid_mask=distance_map<0,colormap='hsv'),psa_result_path+"/distanceimg_psa.bmp")

    
    # o3d.visualization.draw_geometries([pcd],)
    cmd_handle_map = {
        "zrange": get_zrange,
        "remove_selected":remove_select,
        "show_selection":show_selection,
    }
    while True:
        cmd = input(">:")
        if cmd == "exit":
            break
        else:
            handle = cmd_handle_map.get(cmd,None)
            if handle:
                try:
                    handle()
                except Exception as e:
                    print(e.with_traceback())
# %%
