from genericpath import exists
import cv2
import matplotlib.pyplot as plt
import pyaoi
import numpy as np
from pyaoi.structuredlight.utils import PatternOrientation

from pyaoi.utils.daheng import  *

from pyaoi.structuredlight import ThreeFrequencyHeterodyne
from pyaoi.calibration import CameraCalibration,CalibrationData
from constants import *
from utils import *


do_project = True


# 投影图像images并采集
def project_and_capture(images,save_to,confirm=False,camera_id=0):
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("img",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("img",1920,0)


    camera_manager = DaHengCameraManager()
    camera0 = camera_manager.get_camera(camera_id)
    if camera0 is None:
        print("cant open camera %d"%camera_id)
        return False

    camera0.open(camera_exposure,0.0)

    cv2.namedWindow("capture",cv2.WINDOW_NORMAL)
    cv2.moveWindow("capture",0,0)
    cv2.resizeWindow("capture",640,480)

    if confirm:
        white_pattern = np.full_like(images[0],fill_value=255)
        cv2.imshow("img",white_pattern)
        
        while True:
            image_cap = camera0.capture()
            image_cap = pyaoi.to_gray(image_cap)
            cv2.imshow("capture",image_cap)
            key = cv2.waitKey(50)
            if key == 32:
                break
            elif key == 27:
                cv2.destroyWindow("capture")
                return False

    # print("img:",camera0.get_image() is None)
    pyaoi.makedirs(save_to)

    # 先显示第一张
    pattern = images[0]
    cv2.imshow("img",pattern)
    key = cv2.waitKey(200) # 等待显示稳定

    N = len(images)
    for i in range(N):
        # 采集图像
        image_cap = None
        while image_cap is None:
            image_cap = camera0.capture()

        # 提前显示下一张
        if i + 1 < N:   
            pattern = images[i+1]
            cv2.imshow("img",pattern)
        # 处理图像，保存
        image_cap = pyaoi.to_gray(image_cap)
        cv2.imshow("capture",image_cap)
        key = cv2.waitKey(100)
        if key == 27:
            return False

        pyaoi.imwrite(image_cap,f"{save_to}\\{i}.png",compress=True)
        print(f"{i}/{N}",end='\r')
    
    print("")
    camera0.close()
    return True


cycles = [49,42,36]
step  = 16
image_h = pyaoi.imread(data_prepare_root + "\\f_49_42_36_s_16\\hor\\{:04d}.png",range(step*len(cycles)))
image_v = pyaoi.imread(data_prepare_root + "\\f_49_42_36_s_16\\ver\\{:04d}.png",range(step*len(cycles)))
white_pattern = np.full_like(image_h[0],fill_value=255)

calibrate_patterns = image_h + image_v + [white_pattern]
# pyaoi.imshow(calibrate_patterns)



cam_calib = CameraCalibration(cam_config_path)
proj_calib = CameraCalibration(proj_config_path)

print("calibration pattern number:",len(calibrate_patterns))
print("board number required:",cam_calib.board_number_required)


dst_dir = data_root + "\\calib\\cam0"


proj_base_img = np.zeros(proj_calib.image_size,dtype=np.uint8)

pyaoi.makedirs(calib_phase_root)
while cam_calib.calibration_progress < cam_calib.board_number_required: 
    save_to = dst_dir+"\\%d"%cam_calib.calibration_progress

    print("============%d/%d==========="%(cam_calib.calibration_progress+1,cam_calib.board_number_required))
    if do_project:
        if not project_and_capture(calibrate_patterns,save_to,True):
            exit(0)

    images = pyaoi.imread(save_to+"\\{:d}.png",range(len(calibrate_patterns)))
    success,cam_corners,obj_corners = cam_calib.add_image(images[-1])
    print("%s"%("success" if success else "fail"))

    if not success: continue 

    ind = cam_calib.calibration_progress - 1

    # 条纹图像解码
    if True:
        cmd = f"{decode_pattern_exe} -f {','.join([str(x) for x in cycles])} -s {str(step)} -i {save_to} -o {calib_phase_root} --single --offset {0} --suffix _{ind}_h"
        os.system(cmd)
        cmd = f"{decode_pattern_exe} -f {','.join([str(x) for x in cycles])} -s {str(step)} -i {save_to} -o {calib_phase_root} --single --offset {len(cycles)*step} --suffix _{ind}_v"
        os.system(cmd)


    phase_hor = pyaoi.imread(calib_phase_hor.format(ind))
    phase_ver = pyaoi.imread(calib_phase_ver.format(ind))
    mask_hor = pyaoi.imread(calib_phase_mask_hor.format(ind))
    mask_ver = pyaoi.imread(calib_phase_mask_ver.format(ind))

    proj_corners = []
    for cam_point in cam_corners:
        p_hor = calcSubPixelPhase(phase_hor,mask_hor,cam_point[0],cam_point[1])
        p_ver = calcSubPixelPhase(phase_ver,mask_ver,cam_point[0],cam_point[1])
        if p_hor is None or p_ver is None:
            success = False
            break
        else:
            px = p_ver*proj_calib.image_size[1]
            py = p_hor*proj_calib.image_size[0]
        proj_corners.append([px,py])
    if not success:
        print("find projector corner fail")
        cam_calib.remove_last_image()
        continue

    proj_corners = np.array(proj_corners,dtype=np.float32)
    proj_corner_img =  proj_calib.draw_chessboard(proj_base_img,proj_corners)
    
    if do_project:  
        success = pyaoi.imshow(proj_corner_img)

    if not success:
        print("projector corner error")
        cam_calib.remove_last_image()

    proj_calib.add_feature_corresponds(proj_corners,obj_corners)


reproject_error = cam_calib.calibrate()
print("camera reproject_error:",reproject_error)
print(cam_calib.calibration_data)
cam_calib.calibration_data.save(cam_data_path) 


proj_reproj_error = proj_calib.calibrate()
proj_calib.calibration_data.save(proj_data_path)
print("projector reproject_error:",proj_reproj_error)
print(proj_calib.calibration_data)

print("finish")


