#%%
import flirimageextractor
from matplotlib import cm
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn as sns
import logging

from flirSplit import flir_img_split

if __name__ == '__main__':
    imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片')   # 輸入路徑
    # imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\範例圖像\輸入影像')   # 輸入路徑
    # saveImgpath = r'結果存圖\論文\原始影像_熱影像_去背'
    saveImgpath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\範例圖像\輸出影像'

    palettes = [cm.gnuplot2]                        # 影像調色板

    flirSplit = flir_img_split(imgInputpath, palettes)

    for imgPath in flirSplit.getImglist():
        # print(imgPath)
        imgName = os.path.split(imgPath)[-1]
        savePath = os.path.join(saveImgpath, imgName)

        flirRGB, flirHot = flirSplit.separateNP(imgPath)

        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(flirRGB)
        # plt.show()

        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(flirHot)
        # plt.show()

        # addTwoimg = cv2.addWeighted(flirRGB, 1, flirHot, 0.5, 0)

        # plt.imshow(addTwoimg)
        # # plt.imshow(flirHot)
        # plt.show()

        # offset_x = 120
        # offset_y = 45
        
        # plt.imshow(flirRGB)
        # plt.show()



        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
        flirHot = autoNormal*150
        flirHot = np.stack([flirHot, flirHot, flirHot], axis=2)

        rows, cols, _ = flirHot.shape
        rows = int(rows*2)
        cols = int(cols*2)

        flirHot = cv2.resize(flirHot, (cols, rows), interpolation=cv2.INTER_AREA)

        print(flirRGB.shape[0])
        if flirRGB.shape[0] == 1440:
        # RGBrows, RGBcols = flirRGB.shape
        # centerPoint = (int(RGBrows/2), int(RGBcols/2))
        # print(centerPoint)
            offset_x = 140
            offset_y = 45
            zoom = 2

            roi = flirRGB[offset_x : rows+offset_x, offset_y : cols+offset_y]
            roi = flirRGB[offset_x : rows+offset_x, offset_y : cols+offset_y]

            dst = cv2.addWeighted(roi, 1, flirHot, 0.8, 0 ,dtype=cv2.CV_8U)
            flirRGB[offset_x : rows+offset_x, offset_y : cols+offset_y] = dst


        elif flirRGB.shape[0] == 1080:
            offset_x = 100
            offset_y = 50
            zoom = 2

            roi = flirRGB[offset_x : rows+offset_x, offset_y : cols+offset_y]
            roi = flirRGB[offset_x : rows+offset_x, offset_y : cols+offset_y]

            dst = cv2.addWeighted(roi, 1, flirHot, 0.8, 0 ,dtype=cv2.CV_8U)
            flirRGB[offset_x : rows+offset_x, offset_y : cols+offset_y] = dst
            

        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(flirRGB)
        plt.tight_layout()
        plt.show()

        break