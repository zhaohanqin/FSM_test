

import pyaoi
from constants import *
import matplotlib.pyplot as plt
import cv2


peak_limit = 5

tx = 863
distance_fsa = pyaoi.imread(fsa_result_path+"/distancemap_fsa_%.2f.tiff"%peak_limit)


distance_psa = pyaoi.imread(psa_result_path+"/distancemap_psa.tiff")



yslice = slice(279,700)

line_fsa = distance_fsa[yslice,tx]
line_psa = distance_psa[yslice,tx]
plt.plot(line_fsa)
plt.plot(line_psa)
plt.legend(['fsa','psa'])
plt.show()


# pyaoi.imshow(pyaoi.convert_float_image(distance_fsa,invalid_mask=distance_fsa<0, colormap='hsv'),normal_window=True)
