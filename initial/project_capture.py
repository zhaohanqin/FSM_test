"""
自动投影采集
"""

import cv2
# import matplotlib.pyplot as plt
import pyaoi

from pyaoi.utils.daheng import *


# from pyaoi.structuredlight import ThreeFrequencyHeterodyne,PatternOrientation

from constants import *


# 投影图像images并采集
def project_and_capture_general(images,save_to,padto):
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("img",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("img",1920,0)


    cv2.namedWindow("capture",cv2.WINDOW_NORMAL)
    cv2.moveWindow("capture",0,0)
    cv2.resizeWindow("capture",640,480)

    camera_manager = DaHengCameraManager()
    camera0 = camera_manager.get_camera(camera_id)
    camera0.open(camera_exposure,0.0)

    print(save_to)

    pyaoi.makedirs(save_to)


    pattern = images[0]
    cv2.imshow("img",pattern)
    key = cv2.waitKey(200)

    N = len(images)
    for i in range(N):
        pattern = images[i]
        # if pattern.shape != padto:
        #     pad_v = padto[0] - pattern.shape[0]
        #     pad_h = padto[1] - pattern.shape[1]
        #     padtop = pad_v//2
        #     padleft = pad_h //2
        #     pattern = np.pad(pattern,((padtop,pad_v-padtop),(padleft,pad_h-padleft))) 

        image_cap = None
        while image_cap is None:
            image_cap = camera0.capture()

        if i + 1 < N:
            pattern = images[i+1]
            cv2.imshow("img",pattern)

        image_cap = pyaoi.to_gray(image_cap)
        cv2.imshow("capture",image_cap)
        key = cv2.waitKey(100)
        if key == 27:
            break    
        save_path = f"{save_to}/{i}.png"
        pyaoi.imwrite(image_cap,save_path,compress=True)
        print(f"{i}/{N}",end='\r')



        
    print("")
    camera0.close()


def check_project_image(white_pattern):
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("img",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("img",1920,0)

    cv2.namedWindow("capture",cv2.WINDOW_NORMAL)
    cv2.moveWindow("capture",0,0)
    cv2.resizeWindow("capture",640,480)


    cv2.imshow("img",white_pattern)
    
    camera_manager = DaHengCameraManager()
    camera0 = camera_manager.get_camera(camera_id)
    camera0.open(camera_exposure,0.0)

    while True:
        image_cap = camera0.capture()
        image_cap = pyaoi.to_gray(image_cap)
        
        cv2.imshow("capture",image_cap)
        key = cv2.waitKey(50)
        if key == 32:
            break
        elif key == 27:
            break
        
    pyaoi.imwrite(image_cap, "check.png")
    cv2.destroyAllWindows()
    camera0.close()



if __name__ == "__main__":


    dirname = f"f_{'_'.join([str(x) for x in psa_cycles])}_s_{psa_step}"
    print("load pattern...")
    image_h = pyaoi.imread(data_prepare_root + "\\" + dirname + "\\hor\\{:04d}.png",range(psa_step*len(psa_cycles)))
    image_v = pyaoi.imread(data_prepare_root + "\\" + dirname + "\\ver\\{:04d}.png",range(psa_step*len(psa_cycles)))

    project_and_capture_general(image_h,f"{psa_data_path}\\hor",padto=(768,1024))
    project_and_capture_general(image_v,f"{psa_data_path}\\ver",padto=(768,1024))

    exit(0)

    M = 121
    print("load pattern...")
    images_hor = pyaoi.imread(prepare_path_hor+"\{:04d}.bmp",range(M*4))
    images_ver = pyaoi.imread(prepare_path_ver+"\{:04d}.bmp",range(M*4))

    print("project pattern...")
    project_and_capture_general(images_hor,f"{data_root}\\fsa_{obj_name}\\hor",padto=(768,1024))
    project_and_capture_general(images_ver,f"{data_root}\\fsa_{obj_name}\\ver",padto=(768,1024))
