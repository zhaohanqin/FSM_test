
import time
today = time.strftime("%Y%m%d")[2:]


# date = today

date = "220313"

# 相机采集参数
camera_id = 0
camera_exposure_unit = 16666.66
camera_exposure = 16666.66*1

# 投影仪大小
# W = 912
# H = 1140
H = 768
W = 1024


# 主要根路径
code_root = ".\\"
data_prepare_root = r"I:\data-fsa2\\prepare"
data_root = "I:\data-fsa2\\" + date

# 配置文件路径 
cam_config_path = code_root+"\\config\\camera.json"
proj_config_path = code_root+"\\config\\projector.json"


# fsa投影图像路径
prepare_path_hor = data_prepare_root+r"\1024x768\hor"
prepare_path_ver = data_prepare_root+r"\1024x768\ver"



# 实验结果保存根目录
result_root = code_root + f"\\result\\{date}"

# 相机标定结果路径
cam_data_path = result_root+f"\\cam0\\cam_data.xml"
cam_corners_path = result_root+f"\\cam0\\cam_corners.xml"
# 投影仪标定结果路径
proj_corners_path = result_root+f"\\cam0\\proj_corners.xml"
proj_data_path = result_root+f"\\cam0\\proj_data.xml"


# 标定图解码结果目录
calib_phase_root = result_root + "\\cam0\\phases"
calib_phase_mask_hor = calib_phase_root + "\\mask_{}_h.bmp"
calib_phase_mask_ver = calib_phase_root + "\\mask_{}_v.bmp"
calib_phase_hor = calib_phase_root + "\\phase_{}_h.tiff"
calib_phase_ver = calib_phase_root + "\\phase_{}_v.tiff"



"""
可执行文件
"""
executable_root = "D:\\cpp\\insp\\bin"
decode_pattern_exe = executable_root + "\\decode_pattern.exe"
decode_calib_pattern_exe = executable_root + "\\decode_calibrate_pattern.exe"


obj_name = "b"
fsa_data_path = f"{data_root}/fsa_{obj_name}"
fsa_result_path = f"result/{date}/fsa_{obj_name}"


# psa_cycles = [49,42,36]
psa_cycles = [121,110,100]
psa_step = 64
psa_data_path = f"{data_root}/psa_{obj_name}/f_{'_'.join([str(x) for x in psa_cycles])}_s_{psa_step}"
psa_result_path =  f"result/{date}/psa_{obj_name}/f_{'_'.join([str(x) for x in psa_cycles])}_s_{psa_step}"
