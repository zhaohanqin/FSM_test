from constants import *


# %%

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


def get_pptk_selection():
	selected = v.get("selected")
	# selected_location = (point_locations[0][selected],point_locations[1][selected])
	selected_points = point_cloud[selected]
	return selected_points

# def showSelectedPointsInImage():
# 	mask_pattern = cv2.cvtColor(white_pattern,cv2.COLOR_GRAY2BGR)


# 	selected = v.get("selected")
# 	selected_location = (point_locations[0][selected],point_locations[1][selected])
# 	mask_pattern[selected_location[0],selected_location[1]] = (255,0,0)
# 	pyaoi.imshow([mask_pattern])
# 	pyaoi.imwrite(mask_pattern,fsa_result_root+"\\mask_pattern.bmp")

# 	# pyaoi.imwrite([mask_pattern,phase_image,mask_pattern_1,phase_image_1],"image_{}.bmp")

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
    pyaoi.imshow([mask_pattern],normal_window=True)	
    stamp = time.strftime("%Y%m%d%H%M%S")[2:]
    pyaoi.imwrite(mask_pattern,f"{fsa_result_path}/selection_{stamp}.bmp")

def remove_select():
    global point_cloud,point_color
    selected = v.get("selected")
    if len(selected) == 0:
        print("no selection")
        return
    # print(selected)

    point_cloud = np.delete(point_cloud,selected,axis=0)
    point_color = np.delete(point_color,selected,axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(point_color/255)
    stamp = time.strftime("%Y%m%d%H%M%S")[2:]
    o3d.io.write_point_cloud(fsa_result_path+f"/pointcloud_fsa_%.2f_filter_{stamp}.ply"%peak_limit,pcd)


    v.clear()
    v.load(point_cloud)
    v.attributes(point_color/255)
    v.set(point_size=0.01)


if __name__ == "__main__":


	# images = pyaoi.imread(fsa_data_root+"\\hor\image_{}.png",range(300))
	# white_pattern = np.max(np.array(images),axis=0)
    white_pattern = pyaoi.imread(fsa_data_path + "/hor/0.png")
    # pyaoi.imshow(white_pattern,normal_window=True)
    

    oversample = 4
    peak_limit = 0.8

    # max_distance = 787455
    # min_distance = 495641 

    # min_distance,max_distance = 680854.0450982018,724735.2476293959
    max_distance,min_distance = None, None
    # mask
    # mask_hor = pyaoi.imread(fsa_result_dir+"\\mask_hor.bmp")
    # mask_ver = pyaoi.imread(fsa_result_dir+"\\mask_ver.bmp")

    # disparity
    disparity_hor = pyaoi.imread(fsa_result_path+"/disp_hor.tiff")
    disparity_ver = pyaoi.imread(fsa_result_path+"/disp_ver.tiff")

    disparity_hor = np.round(disparity_hor*oversample).astype(np.int32)
    disparity_ver = np.round(disparity_ver*oversample).astype(np.int32)


    # peak
    peak_hor = pyaoi.imread(fsa_result_path+"/peak_hor.tiff")
    peak_ver = pyaoi.imread(fsa_result_path+"/peak_ver.tiff")
    peak_min = np.minimum(peak_hor,peak_ver)


    # peak image
    img_peak_hor = pyaoi.convert_float_image(peak_hor,colormap=cv2.COLORMAP_HSV,invalid_mask=peak_hor<peak_limit)
    img_peak_ver = pyaoi.convert_float_image(peak_ver,colormap=cv2.COLORMAP_HSV,invalid_mask=peak_ver<peak_limit)
    img_peak_min = pyaoi.convert_float_image(peak_min,colormap=cv2.COLORMAP_HSV,invalid_mask=peak_min<peak_limit)
    # pyaoi.imwrite([img_peak_min,img_peak_hor,img_peak_ver],fsa_result_root+"/peak_{}.bmp")
    # pyaoi.imshow([img_peak_min,img_peak_hor,img_peak_ver],normal_window=True)


    # 错误点幅值
    # mask_image = pyaoi.imread(f"{fsa_result_path}/selection_220313203739.bmp",force_gray=False)
    # mask = (mask_image==(255,0,0)).all(axis=2)
    # plt.plot(peak_min[mask].flatten())
    # plt.show()
    # exit(0)

    

    # invalid mask
    invalid_mask =  (peak_min < peak_limit)

    valid_image = pyaoi.to_rgb(white_pattern)
    valid_image[invalid_mask] = (0,0,255)
    pyaoi.imshow(valid_image,normal_window=True)

    # mark invalid disparity
    disparity_hor[invalid_mask] = -1 
    disparity_ver[invalid_mask] = -1 


    # hor_img = pyaoi.convert_float_image(disparity_hor,max_val=resolution[0],min_val=0,invalid_mask=invalid_mask, colormap=cv2.COLORMAP_HSV)
    # ver_img = pyaoi.convert_float_image(disparity_ver,max_val=resolution[1],min_val=0,invalid_mask=invalid_mask, colormap=cv2.COLORMAP_HSV)
    # pyaoi.imshow([hor_img,ver_img])
    # pyaoi.imwrite(hor_img,fsa_result_dir+"\\hor_img.bmp")
    # pyaoi.imwrite(ver_img,fsa_result_dir+"\\ver_img.bmp")

    # tx = 1570
    # ty = 900
    # print(disparity_hor[ty,tx],disparity_ver[ty,tx])
    # print(peak_hor[ty,tx],peak_ver[ty,tx])



    #加载投影仪，相机标定文件
    cam_calib_data = CalibrationData(cam_data_path)
    proj_calib_data = CalibrationData(proj_data_path)

    resolution = proj_calib_data.model_resolution

    # print("camera calib data:\n",cam_calib_data)
    # print("projector calib data:\n",proj_calib_data)



    # geometry = Geometry()
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


    #生成点云
    tstart = time.time()
    point_cloud,point_color,distance_map,point_locations = geometry.generate_pointcloud(view_id,disparity_ver,disparity_hor,white_pattern=white_pattern)
    print("generate cloud time:",time.time()-tstart)


    invalid_distance_mask = distance_map < 0
    # distance_map[distance_map<0] = distance_map.min()
    # pyaoi.imshow(invalid_mask*255)
    # pyaoi.imshow(pyaoi.convert_float_image(distance_map,invalid_mask=invalid_mask, colormap=cv2.COLORMAP_HSV))

    # # 选择错误点
    # mask_image = pyaoi.imread(f"{fsa_result_path}/selection_220313191501.bmp",force_gray=False)
    # mask = (mask_image==(255,0,0)).all(axis=2)
    # ays,axs = np.where(mask)
    # locs = np.array([(point_locations[0][i],point_locations[1][i]) for i in range(len(point_locations[0]))])

    # mask_points = []
    # for ay,ax in zip(ays,axs):
    #     # print()
    #     index = np.where((locs == (ay,ax)).all(axis=1))
    #     mask_points.append(index[0][0])

    # 点云显示
    v = pptk.viewer(point_cloud)
    v.attributes(point_color/255)
    v.set(point_size=0.01)

    # v.set(selected=mask_points)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(point_color/255)
    o3d.io.write_point_cloud(fsa_result_path+"/pointcloud_fsa_%.2f.ply"%peak_limit,pcd)
    pyaoi.imwrite(distance_map,fsa_result_path+"/distancemap_fsa_%.2f.tiff"%peak_limit)
    pyaoi.imwrite(pyaoi.convert_float_image(distance_map,invalid_mask=distance_map<0,colormap='hsv'),fsa_result_path+"/distanceimg_fsa_%.2f.bmp"%peak_limit)
    # pyaoi.imshow(pyaoi.convert_float_image(distance_map,invalid_mask=invalid_distance_mask, colormeap=cv2.COLORMAP_HSV))


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
            
            

    # o3d.visualization.draw_geometries([pcd],)
