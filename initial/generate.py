from distutils.command.config import config
import numpy as np
import matplotlib.pyplot as plt
import pyaoi
from constants import *





def generate_pattern(dir,W,H):
    off,amp = 255/2,255/2
    N = W if dir == 'ver' else H
    R = W if dir == 'hor' else H
    n = np.arange(N)
    images = []
    for w in range(N):
        for i in range(4):
            y = off + amp*np.cos(2*np.pi*w*n/N + np.pi/2*i)
            img = np.round(y).astype(np.uint8)

            if dir == 'ver':
                images.append(img.reshape(1,-1).repeat(R,axis=0))
            else:
                images.append(img.reshape(-1,1).repeat(R,axis=1))

    return images



print("generate pattern ...")
# images_hor = generate_pattern('hor',W,H)
images_ver = generate_pattern('ver',W,H)

tx,ty = 200,200

images_ver = np.array(images_ver)
I0 = images_ver[0::4,ty,tx].astype(np.float32)
I1 = images_ver[1::4,ty,tx].astype(np.float32)
I2 = images_ver[2::4,ty,tx].astype(np.float32)
I3 = images_ver[3::4,ty,tx].astype(np.float32)

Dc = I0 - I2
Ds = I3 - I1

D = Dc + 1j*Ds

C = np.abs(np.fft.fft(D))
plt.plot(C,'-o')
plt.show()

# pyaoi.makedirs(prepare_path_hor)
# pyaoi.makedirs(prepare_path_ver)

# print("save pattern ...")
# pyaoi.imwrite(images_hor,prepare_path_hor+"\{:04d}.bmp")
# pyaoi.imwrite(images_ver,prepare_path_ver+"\{:04d}.bmp")
