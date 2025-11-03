# 从原条纹图进行解码测试

import numpy as np
import matplotlib.pyplot as plt
import pyaoi
from constants import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("read images")
path_ver = r"G:\data-fsa2\prepare\1024x768\ver"


# images_hor = pyaoi.imread(path_hor+"\{:04d}.bmp",range(H*4))
# images = pyaoi.imread(path_ver+"\{:04d}.bmp",range(W*4))
# images = np.array(images)
# np.save("image.npy",images)

images = np.load("image.npy")


tx,ty = 900,20
I0 = images[0::4,ty,tx].astype(np.float32)
I1 = images[1::4,ty,tx].astype(np.float32)
I2 = images[2::4,ty,tx].astype(np.float32)
I3 = images[3::4,ty,tx].astype(np.float32)

Dc = I0 - I2
Ds = I3 - I1

D = Dc + 1j*Ds

C = np.abs(np.fft.fft(D))
plt.plot(C,'-o')
plt.show()







M = 121

win_rect = np.zeros_like(Dc)
win_rect[:M] = 1



win_hamm = np.zeros_like(Dc)
hamm = np.hamming(M)
hamm = (hamm - np.min(hamm))/(np.max(hamm) - np.min(hamm))
win_hamm[:M] = hamm
win_hamm[:M//2] = 1


D_rect = D * win_rect
D_hamm = D *win_hamm


print(D_rect.shape)


plt.figure()
plt.subplot(221)
plt.plot(Dc[:])
plt.plot(Ds[:])
plt.legend(['Dcos','Dsin',])
plt.title("原始信号")

plt.subplot(222)
plt.plot(win_rect)
plt.plot(win_hamm)
plt.legend(['矩形窗','汉明窗',])
plt.title("窗函数")


plt.subplot(223)
plt.plot(np.real(D_rect))
plt.plot(np.imag(D_rect))
plt.legend(['Dcos','Dsin',])
plt.title("乘宽度为%d的矩形窗"%M)

plt.subplot(224)
plt.plot(np.real(D_hamm))
plt.plot(np.imag(D_hamm))
plt.legend(['Dcos','Dsin',])
plt.title("乘宽度为%d的汉明窗"%M)
plt.show()


# 矩形
# 100: 0.97
# 200: 0.89
# 300: 0.767


C_rect = np.abs(np.fft.fft(D_rect))
C_hamm = np.abs(np.fft.fft(D_hamm))

# print(np.sort(C_rect)[-2]/np.sort(C_rect)[-1])
# print(np.sort(C_hamm)[-2]/np.sort(C_hamm)[-1])

plt.plot((C_rect/np.max(C_rect)),'-o')
plt.plot((C_hamm/np.max(C_hamm)),'-o')
plt.legend(['矩形窗','汉明窗',])
plt.title("频谱图")


plt.show()