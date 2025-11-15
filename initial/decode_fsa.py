# 解码图像
# FSA (Frequency Shift Analysis) 频移法解码
# 原理：通过投影一系列频率递增的正弦条纹，使用DFT变换解码出投影仪坐标
from turtle import color, distance
import numpy as np
import matplotlib.pyplot as plt
import pyaoi
from constants import *
import os
from pyaoi.calibration import CameraCalibration,CalibrationData
import cv2

pyaoi.makedirs(fsa_result_path)


M = 121  # 采集的频率数量（M个频率，每个频率4张相移图，共M*4张图像）



ax, ay = 900,495  # 调试用的像素坐标
# ax, ay = 1049,875

debug = False  # 是否启用调试模式

def decode_fsa(dir, frequency_step=1.0):
    """
    FSA频移法解码函数
    
    参数:
    -----
    dir : str
        解码方向，'hor'表示水平条纹（解码投影仪行坐标），'ver'表示垂直条纹（解码投影仪列坐标）
    frequency_step : float
        频率步长（Hz），默认为1.0。当设置为0.5时，频率间隔为0.5Hz，需要更多图像
    
    返回:
    -----
    disparity : np.ndarray
        视差图（投影仪坐标），形状为(height, width)
    peak : np.ndarray
        峰值强度图（置信度），形状为(height, width)
    
    原理说明:
    --------
    1. 四步相移法：
       每个频率投影4张图，相位分别为 0°, 90°, 180°, 270°
       这样可以消除环境光和表面反射率的影响
       
    2. 复数信号构造：
       Dc = I0 - I2  (余弦分量，消除直流分量)
       Ds = I3 - I1  (正弦分量，消除直流分量)
       D = Dc + j*Ds (构造复数信号，便于DFT变换)
       
    3. DFT变换：
       对每个像素点的频率序列进行FFT变换，将时域（频率序列）转换到频域（投影仪坐标域）
       频谱峰值位置即对应投影仪的坐标
       
    4. 峰值检测：
       投影仪坐标 = argmax(|C|)  (找到频谱幅值的最大值位置)
       峰值强度 = max(|C|) / N   (归一化的峰值强度，用于质量评估)
    
    频率步长说明:
    -----------
    - frequency_step = 1.0: 标准步长，频率为 0, 1, 2, ..., (M-1) Hz
      此时频率数量M通常等于投影仪分辨率N，频率索引直接对应投影仪坐标
      
    - frequency_step = 0.5: 精细步长，频率为 0, 0.5, 1.0, ..., (M-1)*0.5 Hz
      此时频率数量M = 2*N，需要将频率索引映射到投影仪坐标
      映射公式：投影仪坐标 = 频率索引 * (投影仪分辨率 / 频率数量)
    """
    cache_path = f"{fsa_result_path}/images_{dir}.npy"

    # N是投影仪的分辨率（水平方向用W，垂直方向用H）
    # 注意：当frequency_step=0.5时，M != N，需要后续映射
    N = H if dir == "hor" else W
    
    # 加载图像序列
    # 图像组织方式：[freq0_0°, freq0_90°, freq0_180°, freq0_270°, freq1_0°, freq1_90°, ...]
    # 共M个频率，每个频率4张相移图，总共M*4张图像
    if not os.path.exists(cache_path):
        print("load images...")
        images = np.array(pyaoi.imread(fsa_data_path+"/"+dir+"/{:d}.png",range(M*4)))
        np.save(cache_path,images)
    else:
        images = np.load(cache_path)

    images = images.astype(np.float32)
    
    # 步骤1: 提取四个相移图像序列
    # 从按频率组织的图像序列中，重新组织为按相位分组
    # I0: 所有频率的0°相位图像，形状为(M, height, width)
    I0 = images[0::4]  # 提取索引 0, 4, 8, 12, ... (所有0°图像)
    I1 = images[1::4]  # 提取索引 1, 5, 9, 13, ... (所有90°图像)
    I2 = images[2::4]  # 提取索引 2, 6, 10, 14, ... (所有180°图像)
    I3 = images[3::4]  # 提取索引 3, 7, 11, 15, ... (所有270°图像)

    # 步骤2: 构造复数信号
    # Dc = I0 - I2: 余弦分量（实部），通过相减消除直流分量和环境光
    # Ds = I3 - I1: 正弦分量（虚部），通过相减消除直流分量和环境光
    # D = Dc + j*Ds: 构造复数信号，便于进行DFT变换
    # 这个复数信号包含了每个频率下的相位信息
    Dc = I0 - I2
    Ds = I3 - I1
    D = Dc + 1j*Ds  # 复数信号，形状为(M, height, width)

    if debug:
        # 调试：绘制某个像素点的强度序列
        plt.plot(I0[:,ay,ax], label='I0 (0°)')
        plt.plot(I2[:,ay,ax], label='I2 (180°)')
        plt.plot(Dc[:,ay,ax], label='Dc (I0-I2)')
        plt.legend()
        plt.xlabel('频率索引')
        plt.ylabel('强度')
        plt.title(f'像素点({ax},{ay})的强度序列')
        plt.show()

    # 初始化输出数组
    disparity = np.empty(D.shape[1:],dtype=np.float32)  # 视差图（投影仪坐标）
    peak = np.empty(D.shape[1:],dtype=np.float32)      # 峰值强度图（置信度）

    # 步骤3: 分块DFT处理
    # 分块处理可以降低内存占用，特别是对于大图像
    block_size = 100  # 分块大小
    t = 0   
    total = ((D.shape[1]+block_size-1)//block_size) * ((D.shape[2]+block_size-1)//block_size)

    print(f"decode {dir}:")
    print(f"  频率数量 M = {M}")
    print(f"  投影仪分辨率 N = {N}")
    print(f"  频率步长 = {frequency_step} Hz")
    if frequency_step != 1.0:
        print(f"  注意：频率步长为{frequency_step}，需要将频率索引映射到投影仪坐标")
        coord_scale = N / M  # 坐标映射比例
        print(f"  坐标映射比例 = {coord_scale:.4f}")
    
    for ty in range(0,D.shape[1],block_size):
        for tx in range(0,D.shape[2],block_size):
            # 当前块的范围
            ey = min(ty+block_size,D.shape[1])
            ex = min(tx+block_size,D.shape[2])
            
            # 步骤3.1: 执行DFT变换
            # np.fft.fft(D_block, n=N, axis=0)
            # - D_block: 形状为(M, block_h, block_w)，包含当前块所有像素点的频率序列
            # - n=N: DFT长度。当frequency_step=1.0时，N=M，频率索引直接对应投影仪坐标
            #         当frequency_step=0.5时，M=2*N，需要使用M作为DFT长度，然后映射到投影仪坐标
            # - axis=0: 沿频率维度进行FFT变换
            # - 输出C: 形状为(N, block_h, block_w)，包含频谱信息
            
            # 【关键改进点1】当frequency_step=0.5时，应该使用M作为DFT长度，而不是N
            # 这样可以充分利用所有频率信息，提高频率分辨率
            if frequency_step == 1.0:
                # 标准情况：频率数量M等于投影仪分辨率N，直接使用N作为DFT长度
                fft_length = N
            else:
                # 频率步长不为1.0时：使用频率数量M作为DFT长度，以获得更高的频率分辨率
                fft_length = M
            
            # 执行FFT变换
            C = np.abs(np.fft.fft(D[:,ty:ey,tx:ex], n=fft_length, axis=0))
            # C的形状为(fft_length, block_h, block_w)
            # C[k]表示频率索引为k时的频谱幅值
            
            # 步骤3.2: 峰值检测
            # 找到每个像素点频谱幅值的最大值位置，即投影仪坐标
            # argmax返回的是频率索引，需要根据frequency_step映射到投影仪坐标
            freq_index = np.argmax(C, axis=0)  # 频率索引，形状为(block_h, block_w)
            
            # 【关键改进点2】当frequency_step=0.5时，需要将频率索引映射到投影仪坐标
            if frequency_step == 1.0:
                # 标准情况：频率索引直接等于投影仪坐标
                disparity[ty:ey,tx:ex] = freq_index
            else:
                # 频率步长不为1.0时：需要映射
                # 映射公式：投影仪坐标 = 频率索引 * (投影仪分辨率 / 频率数量)
                coord_scale = N / M
                disparity[ty:ey,tx:ex] = freq_index * coord_scale
            
            # 步骤3.3: 计算峰值强度（归一化）
            # 峰值强度用于评估解码质量，值越大表示该点的解码质量越好
            # 归一化：除以DFT长度，使得峰值强度在[0, 1]范围内
            peak[ty:ey,tx:ex] = np.max(C, axis=0) / fft_length

            # if tx >= ax and ex < ax and ty >= ay and ey < ay:

            
            t += 1
            print("\r%d/%d"%(t,total),end="")

    print("")
    return disparity,peak



def deal_one(dir, frequency_step=1.0):
    """
    处理单个方向的解码结果
    
    参数:
    -----
    dir : str
        解码方向，'hor'或'ver'
    frequency_step : float
        频率步长（Hz），默认为1.0
    """
    disp_path = f"{fsa_result_path}/disp_{dir}.tiff"
    peak_path = f"{fsa_result_path}/peak_{dir}.tiff"
    if not os.path.exists(disp_path):
        # 执行解码
        disparity,peak = decode_fsa(dir, frequency_step=frequency_step)
        # 保存结果
        pyaoi.imwrite(disparity,disp_path)
        pyaoi.imwrite(peak,peak_path)
    else:
        # 如果结果已存在，直接加载
        disparity,peak = pyaoi.imread(disp_path),pyaoi.imread(peak_path)
    
    # 输出调试点的解码结果
    print(f"点({ax},{ay})的解码结果: 投影仪坐标={disparity[ay,ax]:.2f}, 峰值强度={peak[ay,ax]:.4f}")

    # 阈值过滤：峰值强度低于阈值的点被认为是无效点
    thres = 1
    # 将视差图转换为伪彩色图像，无效点用mask标记
    disp = pyaoi.convert_float_image(disparity,colormap="hsv",invalid_mask=peak < thres)
    # 将峰值强度图转换为伪彩色图像，无效点用mask标记
    peak = pyaoi.convert_float_image(peak,colormap="hsv",invalid_mask=peak < thres)
    
    if debug:
        # 调试模式下显示结果
        pyaoi.imshow([disp,peak],normal_window=True)

    # 保存可视化结果
    pyaoi.imwrite(disp,f"{fsa_result_path}/disp_{dir}.bmp")
    pyaoi.imwrite(peak,f"{fsa_result_path}/peak_{dir}.bmp")

def main(frequency_step=1.0):
    """
    主函数：解码水平和垂直两个方向
    
    参数:
    -----
    frequency_step : float
        频率步长（Hz），默认为1.0
    """
    deal_one("hor", frequency_step=frequency_step)
    deal_one("ver", frequency_step=frequency_step)
    


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













