import numpy as np

"""
计算亚像素相位值
"""
def calcSubPixelPhase(phase,mask,x,y):
    cx = int(np.floor(x))                      
    cy = int(np.floor(y))

    p_left = phase[cy,cx-1]
    p_right = phase[cy,cx+1]
    p_top = phase[cy-1,cx]
    p_bot = phase[cy+1,cx]

    if mask[cy,cx-1] == 0 or mask[cy,cx+1] == 0 or mask[cy-1,cx] == 0 or mask[cy+1,cx] == 0:
        print("nerber error")
        return

    dpx = (p_right - p_left)/2
    dpy = (p_bot - p_top)/2

    dx = x - cx
    dy = y - cy

    p = phase[cy,cx] + dpx*dx + dpy*dy
    return p