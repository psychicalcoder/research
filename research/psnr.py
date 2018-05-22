from PIL import Image
import numpy as np
import math
import glob
import os
import matplotlib.pyplot as plt

WIDTH = 256
HEIGHT = 256
CHANNELS = 3

def get_image(zip_path, hr_path):
    lr = Image.open(zip_path)
    hr = Image.open(hr_path)
    return np.array(lr), np.array(hr)

def psnr(target, ref):
    R = target[:,:,0]-ref[:,:,0]
    G = target[:,:,1]-ref[:,:,1]
    B = target[:,:,2]-ref[:,:,2]
    mser = R*R
    mseg = G*G
    mseb = B*B
    SUM = mser.sum()+mseg.sum()+mseb.sum()
    MSE = SUM / (WIDTH*HEIGHT*CHANNELS)
    PNSR = 10*math.log(((255. * 255.)/(MSE)), 10)
    return PNSR


if __name__ == "__main__":
    cwd1 = "./Org/IMG_1.png"
    cwd2 = "./TEST75/IMG_1.png"
    a, b = get_image(cwd1, cwd2)
    p = psnr(a, b)
    print(p)
