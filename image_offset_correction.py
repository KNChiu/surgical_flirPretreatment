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


def resize_flirHot(flirHot_original, zoom):
    '''自訂函數 : 縮放熱影像'''

    flirHot = flirHot_original.copy() 
    autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
    flirHot = autoNormal*180                                        # 影像強度
    flirHot = np.stack([flirHot, flirHot, flirHot], axis=2)         # 疊回3通道

    rows, cols, _ = flirHot.shape                                               
    rows = int(rows*zoom)       # 熱影像縮放倍數
    cols = int(cols*zoom)
    flirHot = cv2.resize(flirHot, (cols, rows), interpolation=cv2.INTER_AREA)
    return flirHot, rows, cols

def calculate_centerPoint(flirRGB, flirHot):
    '''自訂函數 : 計算中心點與偏移量'''

    RGBrows, RGBcols, _ = flirRGB.shape
    centerPointRGB = (int(RGBcols/2), int(RGBrows/2))
    # print(centerPointRGB)
    Hotrows, hotcols, _ = flirHot.shape
    centerPointHot = (int(hotcols/2), int(Hotrows/2))
    correction = (centerPointRGB[0]-centerPointHot[0], centerPointRGB[1]-centerPointHot[1])
    return correction, centerPointRGB, centerPointHot

def draw_centerPoint(flirRGB, flirHot):
    '''自訂函數 : 繪製中心點'''

    correction, centerPointRGB, centerPointHot = calculate_centerPoint(flirRGB, flirHot)
    cv2.circle(flirRGB, centerPointRGB, 20, (255, 0, 0), -1)
    cv2.circle(flirHot, centerPointHot, 30, (0, 0, 255), -1)


def draw_overlay_images(flirRGB, flirHot, offset_x, offset_y):
    '''自訂函數 : 繪製重疊影像'''

    draw_centerPoint(flirRGB, flirHot)
    img = flirRGB.copy()
    roi = img[offset_x : rows+offset_x, offset_y : cols+offset_y]
    dst = cv2.addWeighted(roi, 1, flirHot, 0.8, 0 ,dtype=cv2.CV_8U)
    img[offset_x : rows+offset_x, offset_y : cols+offset_y] = dst

    return img



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
        saveImgpath = saveImgpath + '//correction_flir'
        savePath = os.path.join(saveImgpath, imgName)

        flirRGB, flirHot_original = flirSplit.separateNP(imgPath)                               # 分離影像
        
        flirHot, rows, cols = resize_flirHot(flirHot_original, zoom = 1.9)                      # 縮放熱影像
        
        correction, centerPointRGB, centerPointHot = calculate_centerPoint(flirRGB, flirHot)    # 計算中心點與偏移量

        offset_x = 0
        offset_y = 0
        original = draw_overlay_images(flirRGB, flirHot, offset_x, offset_y)                    # 原點重疊

        offset_x = correction[1]
        offset_y = correction[0]
        centers = draw_overlay_images(flirRGB, flirHot, offset_x, offset_y)                     # 中心點重疊
        

        offset_x = correction[1] + 50
        offset_y = correction[0]
        correct_img = draw_overlay_images(flirRGB, flirHot, offset_x, offset_y)                 # 中心點偏移校正

        

        fig = plt.figure()
        subplot1=fig.add_subplot(1, 5, 1)
        subplot1.imshow(flirRGB)
        subplot1.set_title("Visual image")
        subplot1.axis('off')

        subplot2=fig.add_subplot(1, 5, 2)
        subplot2.imshow(flirHot_original, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")
        subplot2.axis('off')

        subplot3=fig.add_subplot(1, 5, 3)
        subplot3.imshow(original)
        subplot3.set_title("Overlapping")
        subplot3.axis('off')

        subplot4=fig.add_subplot(1, 5, 4)
        subplot4.imshow(centers)
        subplot4.set_title("Center")
        subplot4.axis('off')

        subplot5=fig.add_subplot(1, 5, 5)
        subplot5.imshow(correct_img)
        subplot5.set_title("Correction")
        subplot5.axis('off')

        # fig.suptitle(figTitle)
        fig.tight_layout()
        

        savePath = None
        if savePath:
            print("save at:"+ str(savePath))
            fig.savefig(savePath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        plt.show()


        # break