from constants import *


import os
import pyaoi



"""
decode psa pattern
"""
# cycles = [121,110,100]
# step = 64

# cycles = [49,42,36]
# step = 16

pyaoi.makedirs(psa_result_path)

def get_cmd(pat_dir):
    return f"{decode_pattern_exe} -f {','.join([str(x) for x in psa_cycles])} -s {str(psa_step)} -i {psa_data_path}/{'hor' if pat_dir=='h' else 'ver'} -o {psa_result_path} --single --suffix _{pat_dir}"


os.system(get_cmd('h'))
os.system(get_cmd('v'))