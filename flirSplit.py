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

logging.basicConfig(level=logging.ERROR)          # 避免出現WARNINIG


class flir_img_split:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        pass

    def getImglist(self):                                                           # 遍歷路徑
        imgPathlist = []
        for path, dir_list, file_list in self.imgPath: 
            for file_name in file_list:
                if file_name.split(".")[-1] == "jpg":
                    imgPathlist.append(os.path.join(str(path), str(file_name)))
        
        return imgPathlist

    def separateNP(self, imgPath):     # 分離原始圖像與溫度影像
        flir = flirimageextractor.FlirImageExtractor(palettes=self.palettes)                       # 熱影像轉換套件
        flir.process_image(imgPath)       
        flirRGB = flir.extract_embedded_image()                                                     # 輸出 RGB
        flirHot = flir.get_thermal_np()                                                             # 輸出熱影像資訊
        
        return flirRGB, flirHot




if __name__ == '__main__':
    imgPath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片\Ischemia FLIR')   # 輸入路徑

    flirSplit = flir_img_split(imgPath)
    flirSplit.getImglist()

