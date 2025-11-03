# 解码图像
from turtle import color, distance
import numpy as np
import matplotlib.pyplot as plt
import pyaoi
from constants import *
import os
from pyaoi.calibration import CameraCalibration,CalibrationData
import cv2

pyaoi.makedirs(fsa_result_path)


M = 121



ax, ay = 900,495
# ax, ay = 1049,875

debug = False

def decode_fsa(dir):
    cache_path = f"{fsa_result_path}/images_{dir}.npy"

    N = H if dir == "hor" else W
    if not os.path.exists(cache_path):
        print("load images...")
        images = np.array(pyaoi.imread(fsa_data_path+"/"+dir+"/{:d}.png",range(M*4)))
        np.save(cache_path,images)
    else:
        images = np.load(cache_path)

    images = images.astype(np.float32)
    I0 = images[0::4]
    I1 = images[1::4]
    I2 = images[2::4]
    I3 = images[3::4]

    Dc = I0 - I2
    Ds = I3 - I1
    D = Dc + 1j*Ds

    if debug:
        plt.plot(I0[:,ay,ax])
        plt.plot(I2[:,ay,ax])
        plt.plot(Dc[:,ay,ax])
        plt.show()


    disparity = np.empty(D.shape[1:],dtype=np.float32)
    peak = np.empty(D.shape[1:],dtype=np.float32)

    # 解出整张图的最大最小值
    block_size = 100
    t = 0   
    total = ((D.shape[1]+block_size-1)//block_size) * ((D.shape[2]+block_size-1)//block_size)


    print(f"decode {dir}:")
    for ty in range(0,D.shape[1],block_size):
        for tx in range(0,D.shape[2],block_size):
            ey = min(ty+block_size,D.shape[1])
            ex = min(tx+block_size,D.shape[2])
                
            C = np.abs(np.fft.fft(D[:,ty:ey,tx:ex],N,axis=0))
            disparity[ty:ey,tx:ex] = np.argmax(C,axis=0)
            peak[ty:ey,tx:ex] = np.max(C,axis=0)/N

            # if tx >= ax and ex < ax and ty >= ay and ey < ay:

            
            t += 1
            print("\r%d/%d"%(t,total),end="")

    print("")
    return disparity,peak



def deal_one(dir):
    disp_path = f"{fsa_result_path}/disp_{dir}.tiff"
    peak_path = f"{fsa_result_path}/peak_{dir}.tiff"
    if not os.path.exists(disp_path):
        disparity,peak = decode_fsa(dir)
        pyaoi.imwrite(disparity,disp_path)
        pyaoi.imwrite(peak,peak_path)
    else:
        disparity,peak = pyaoi.imread(disp_path),pyaoi.imread(peak_path)
    
    print(disparity[ay,ax],peak[ay,ax])

    thres = 1
    disp = pyaoi.convert_float_image(disparity,colormap="hsv",invalid_mask=peak < thres)
    peak = pyaoi.convert_float_image(peak,colormap="hsv",invalid_mask=peak < thres)
    
    if debug:
        pyaoi.imshow([disp,peak],normal_window=True)

    pyaoi.imwrite(disp,f"{fsa_result_path}/disp_{dir}.bmp")
    pyaoi.imwrite(peak,f"{fsa_result_path}/peak_{dir}.bmp")

def main():
    deal_one("hor")
    deal_one("ver")
    


def fix_peak():
    # dir = "hor"


    dir = "ver"
    N = H if dir == "hor" else W

    cache_path = f"{fsa_result_path}/images_{dir}.npy"

    if not os.path.exists(cache_path):
        print("load images...")
        images = pyaoi.imread(fsa_data_path+"/"+dir+"/{:d}.png",range(M*4))
        np.save(cache_path,np.array(images))
    else:
        images = np.load(cache_path)

    images = images.astype(np.float32)

    I0 = images[0::4]
    I1 = images[1::4]
    I2 = images[2::4]
    I3 = images[3::4]

    Dc = I0 - I2
    Ds = I3 - I1
    D = Dc + 1j*Ds


    source_disp = pyaoi.imread(fsa_result_path+f"/disp_{dir}.tiff")
    source_peak = pyaoi.imread(fsa_result_path+f"/peak_{dir}.tiff")

    mask_image = pyaoi.imread(f"{fsa_result_path}/selection_220314160342.bmp",force_gray=False)

    mask = (mask_image==(255,0,0)).all(axis=2)
    ays,axs = np.where(mask)

    for ay,ax in zip(ays,axs):

        src_disp = int(source_disp[ay,ax])
        C = np.abs(np.fft.fft(D[:,ay,ax],N))/N

        # print(f"=={ay},{ax}===")
        # print("raw:",src_disp,source_peak[ay,ax])
        
        for i,value in enumerate(C):
            if value < 0.8:
                continue

            left_peak = (i == 0) or (C[i] > C[i-1])
            right_peak = (i == len(C)-1) or (C[i] > C[i+1])
            is_peak = left_peak and right_peak
            
            if is_peak:
                if i != src_disp:
                    source_disp[ay,ax] = i
                    print(i,value)
    


    pyaoi.imwrite(source_disp,fsa_result_path+f"/disp_{dir}_fix.tiff")




def test():
    dir = "ver"
    N = H if dir == "hor" else W

    cache_path = f"{fsa_result_path}/images_{dir}.npy"

    if not os.path.exists(cache_path):
        print("load images...")
        images = pyaoi.imread(fsa_data_path+"/"+dir+"/{:d}.png",range(M*4))
        np.save(cache_path,np.array(images))
    else:
        images = np.load(cache_path)

    images = images.astype(np.float32)

    # pyaoi.imshow(images[400].astype(np.uint8))
    
    I0 = images[0::4]
    I1 = images[1::4]
    I2 = images[2::4]
    I3 = images[3::4]

    Dc = I0 - I2
    Ds = I3 - I1
    D = Dc + 1j*Ds

    C = np.abs(np.fft.fft(D[:,ay,ax],N))/N
    # max_index = np.argmax(C)

    # Cs = [(C[i],i) for i in range(len(C))]
    # Cs.sort(reverse=True)
    # print(Cs[:5])


    # print(max_index,C[max_index])
    # disp = pyaoi.imread(f"{fsa_result_path}/disp_ver.tiff")
    # peak = pyaoi.imread(f"{fsa_result_path}/disp_ver.tiff")
    # print(disp[ay,ax],peak[ay,ax])

    proj_calib_data = CalibrationData(proj_data_path)
    ver_phase_path = psa_result_path + "\\phase_v.tiff"
    phase_ver = pyaoi.imread(ver_phase_path)
    disparity_ver = phase_ver*proj_calib_data.model_resolution[1]

    # 画出平铺曲线
    # plt.subplot(211)
    # plt.plot(I0[:,ay,ax])
    # plt.plot(I2[:,ay,ax])
    # plt.plot(Dc[:,ay,ax])
    # plt.legend(['I0','I2','Dc'])
    # plt.subplot(212)


    plt.plot(C)
    plt.vlines(disparity_ver[ay,ax],0,np.max(C),color='red')
    plt.xlabel("$c_p$")
    plt.ylabel("$|C(c_p)|$")
    plt.show()

    exit(0)



def find_peaks(C,limit):
    peaks = []
    for i,value in enumerate(C):
        if value < limit:
            continue
        left_peak = (i == 0) or (C[i] > C[i-1])
        right_peak = (i == len(C)-1) or (C[i] > C[i+1])
        is_peak = left_peak and right_peak
        if is_peak:
            if len(peaks) > 0 and i - peaks[-1] < 5:
                ki = value/(value+C[peaks[-1]])

                peaks[-1] = ki*i+(1-ki)*peaks[-1]
            else:
                peaks.append(i)

    return np.array(peaks)

def get_peaks_position(ays,axs,dir,limit):

    N = H if dir == "hor" else W
    cache_path = f"{fsa_result_path}/images_{dir}.npy"

    if not os.path.exists(cache_path):
        print("load images...")
        images = pyaoi.imread(fsa_data_path+"/"+dir+"/{:d}.png",range(M*4))
        np.save(cache_path,np.array(images))
    else:
        images = np.load(cache_path)

    images = images.astype(np.float32)
    
    I0 = images[0::4]
    I1 = images[1::4]
    I2 = images[2::4]
    I3 = images[3::4]

    Dc = I0 - I2
    Ds = I3 - I1
    D = Dc + 1j*Ds

    result = []
    for y,x in zip(ays,axs):
        C = np.abs(np.fft.fft(D[:,y,x],N))/N



        peaks = find_peaks(C,limit)
        result.append(peaks)

        if debug:
            print(peaks)
            plt.plot(C)
            plt.title(f"dir={dir},x={x},y={y}")
            plt.show()

    return result



def fix_fsa_disparity():

    proj_calib_data = CalibrationData(proj_data_path)

    # mask_image = pyaoi.imread(f"{fsa_result_path}/selection_220314160342.bmp",force_gray=False)
    # pyaoi.imshow(mask_image,normal_window=True)
    # mask = (mask_image==(255,0,0)).all(axis=2)
    # ays,axs = np.where(mask)

    # white_pattern = pyaoi.imread(fsa_data_path + "/hor/0.png")
    

    ays = [ay]  #相机上需要纠正的点
    axs = [ax]


    # 原disp
    raw_disp_hor = pyaoi.imread(fsa_result_path+f"/disp_hor.tiff")
    raw_disp_ver = pyaoi.imread(fsa_result_path+f"/disp_ver.tiff")


    print("get all peaks position")
    peaks_hor = get_peaks_position(ays,axs,"hor",0.5) # 相机上每个像素点在投影仪上的可选y坐标
    peaks_ver = get_peaks_position(ays,axs,"ver",0.5) # 可选x坐标
    print("hor:",peaks_hor)
    print("ver:",peaks_ver)


    # 排列组合得到所有点
    for i in range(len(ays)):

        cy = ays[i]    # 相机像素点
        cx = axs[i]

        lp = get_epipolar_line(cx,cy) # 投影仪上的极线

        pk_hor = peaks_hor[i] # 水平条纹可选峰值
        pk_ver = peaks_ver[i] # 垂直条纹可选峰值
        
        # 所有可选坐标组合
        pys = np.repeat(pk_hor,pk_ver.shape[0])
        pxs = np.repeat(pk_ver.reshape(1,-1),pk_hor.shape[0],axis=0).flatten()
        print("pys:",pys)
        print("pxs:",pxs)

        proj_points = np.c_[pxs,pys]
        undis_points = cv2.undistortPoints(proj_points,proj_calib_data.intrinsic,proj_calib_data.distortion,P=proj_calib_data.intrinsic)
        undis_points.shape = (-1,2)
        distance = []
        for p in undis_points:
            d = np.abs(lp[0]*p[0] + lp[1]*p[1] + lp[2])/np.sqrt(lp[0]**2+lp[1]**2)
            distance.append(d)
        
        print(distance)

        


        raw_py = int(raw_disp_hor[ay,ax])
        raw_px = int(raw_disp_ver[ay,ax])
        print("raw disp:",raw_py,raw_px)



        proj_image = np.zeros((H,W,3),dtype=np.uint8)
        draw_epipolar(proj_image, lp)

        # 画出所有可选点
        proj_image[undis_points[:,1].astype(np.int64),undis_points[:,0].astype(np.int64)] = (255,0,0)
        proj_image[raw_py,raw_px] = (0,255,0)
        pyaoi.imshow(proj_image)

        hs = 40
        proj_image_roi = proj_image[raw_py-hs:raw_py+hs,raw_px-hs:raw_px+hs]
        proj_image_roi_resize = cv2.resize(proj_image_roi,(500,500),interpolation=cv2.INTER_AREA)
        pyaoi.imshow(proj_image_roi_resize)



def draw_epipolar(image,lp):
    x0 = 0
    y0 = -lp[2]/lp[1]

    x1 = image.shape[1]
    y1 = -(lp[2]+lp[0]*x1)/lp[1]

    cv2.line(image,np.array((x0,y0),dtype=np.int64),np.array((x1,y1),dtype=np.int64),(0,0,255),1)

def get_epipolar_line(cx,cy):
    proj_calib_data = CalibrationData(proj_data_path)
    cam_calib_data = CalibrationData(cam_data_path)

    

    Rc,_ = cv2.Rodrigues(cam_calib_data.rotation)
    tc = cam_calib_data.translation
    Kc = cam_calib_data.intrinsic

    Pc = np.empty((3,4),dtype=np.float64)
    Pc[:,:3] = Rc
    Pc[:,-1] = tc
    Pc = Kc@Pc

    Rp,_ = cv2.Rodrigues(proj_calib_data.rotation)
    tp = proj_calib_data.translation
    Kp = proj_calib_data.intrinsic

    Pp = np.empty((3,4),dtype=np.float64)
    Pp[:,:3] = Rp
    Pp[:,-1] = tp
    Pp = Kp@Pp

    camera_center = -Rc.T@tc
    if debug: print("cam center:",camera_center)

    ep = Kp@(Rp@camera_center+tp)
    # ep =Pp@camera_center
    ep = ep/ep[-1]
    if debug: print("ep",ep)

    xc_undis = cv2.undistortPoints(np.array([cx,cy],dtype=np.float64),Kc,cam_calib_data.distortion,P=Kc)[0][0]
    xc = np.array([xc_undis[0],xc_undis[1],1],dtype=np.float64)
    if debug: print("xc:",xc)

    xp = Pp@np.linalg.pinv(Pc)@xc
    xp = xp/xp[-1]
    if debug: print("xp:",xp)

    lp = np.cross(ep,xp)
    lp = lp/lp[-1]
    if debug: print("lp",lp)

    # if debug:
    #     # 标定图解码结果目录
    #     calib_phase_root = result_root + "\\cam0\\phases"
    #     calib_phase_hor = calib_phase_root + "\\phase_{}_h.tiff"
    #     calib_phase_ver = calib_phase_root + "\\phase_{}_v.tiff"

    #     # 从标定的相位图获取(ax,ay)的像素点在投影仪上的位置
    #     calib_phase_hor = pyaoi.imread(calib_phase_hor.format(0))
    #     calib_phase_ver = pyaoi.imread(calib_phase_ver.format(0))

    #     py = calib_phase_hor[ay,ax]*proj_calib_data.model_resolution[0]
    #     px = calib_phase_ver[ay,ax]*proj_calib_data.model_resolution[1]

    #     # print(ax,ay)
    #     print("pxpy:",px,py)

    #     x0 = 0
    #     y0 = -lp[2]/lp[1]

    #     x1 = W
    #     y1 = -(lp[2]+lp[0]*x1)/lp[1]

    #     proj_image = np.zeros((H,W,3),dtype=np.uint8)
    #     cv2.line(proj_image,np.array((x0,y0),dtype=np.int64),np.array((x1,y1),dtype=np.int64),(0,0,255),1)
    #     proj_image[int(py),int(px)] = (0,255,0)
    #     pyaoi.imshow(proj_image)

    return lp



# get_epipolar_line(ax,ay)

test()

# fix_fsa_disparity()


# fix_peak()


# main()













