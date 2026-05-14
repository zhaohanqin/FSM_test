# 解码测试

import numpy as np
import matplotlib.pyplot as plt
import pyaoi
from constants import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("read images")
path_hor = r"G:\data-fsa2\220311\fsa_a\hor"

M = 121
# images_hor = pyaoi.imread(path_hor+"\{:d}.png",range(M*4))
# np.save("result/220311/images_hor.npy",np.array(images_hor))
images_hor = np.load("result/220311/images_hor.npy")



tx,ty = 1787,942
I0 = images_hor[0::4,ty,tx].astype(np.float32)
I1 = images_hor[1::4,ty,tx].astype(np.float32)
I2 = images_hor[2::4,ty,tx].astype(np.float32)
I3 = images_hor[3::4,ty,tx].astype(np.float32)

plt.figure()
plt.subplot(221)
plt.plot(I0)
plt.title("I0")
plt.subplot(222)
plt.plot(I1)
plt.title("I1")

plt.subplot(223)
plt.plot(I2)
plt.title("I2")


plt.subplot(224)
plt.plot(I3)
plt.title("I3")



Dc = np.zeros(H)
Ds = np.zeros(H)
Dc[:M] = I0 - I2
Ds[:M] = I3 - I1

plt.figure()
plt.subplot(211)
plt.plot(Dc[:200])
plt.title("Dc")

plt.subplot(212)
plt.plot(Ds[:200])
plt.title("Ds")



C = np.abs(np.fft.fft(Dc+1j*Ds))
C = C/np.max(C)

c = np.arange(len(C))

max_index = np.argmax(C)
reg_left = max(0,max_index-100)
reg_right = max(0,max_index+100)

print(np.argmax(C))
plt.figure()
plt.plot(c[reg_left:reg_right], C[reg_left:reg_right],'-o')
plt.title("频谱图")

plt.show()
