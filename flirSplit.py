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
    def __init__(self, imgPath, palettes):
        self.imgPath = imgPath
        self.palettes = palettes
        pass

    def getImglist(self):                           # 取得每個影像路徑
        imgPathlist = []
        for path, dir_list, file_list in self.imgPath:                               
            for file_name in file_list:
                if file_name.split(".")[-1] == "jpg":                                               # 只保留副檔名為*.jpg的影像
                    imgPathlist.append(os.path.join(str(path), str(file_name)))
                
        return imgPathlist

    def separateNP(self, imgPath):     # 分離原始圖像與溫度影像
        flir = flirimageextractor.FlirImageExtractor(palettes=self.palettes)                        # 熱影像轉換套件
        flir.process_image(imgPath)       
        flirRGB = flir.extract_embedded_image()                                                     # 輸出 RGB
        flirHot = flir.get_thermal_np()                                                             # 輸出熱影像資訊
        
        return flirRGB, flirHot
    
    def makeMask(self, flirHot):                                        # 圈出溫差 N度內範圍 
        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
        flirMean = flirHot.mean()                                                                 # 計算整張熱影像平均                 
        
        flimask = flirHot.copy()        
        flimask[flirHot < flirMean] = 0                                                             # 產生遮罩

        normalObject = autoNormal.copy()                                                            # 二質化
        normalObject[flimask < flirMean] = 0

        hotObject = flirHot.copy()
        hotObject[flimask < 255] = 0

        return flimask, normalObject, hotObject

    def flirframe_distribution(self, flimask, confidence):
        x = flimask[flimask > 0]                        # 去除為0資料
        x = x.flatten()                                 # 攤平數據
        mean, std = x.mean(), x.std(ddof=1)             # 計算均值與標準差
        conf_intveral = stats.norm.interval(confidence, loc=mean, scale=std)        # 取得信心區間 
        print(str(confidence*100) + "%信賴區間", conf_intveral)


        # flirHot[flimask < 255] = 0
        x = flimask[flimask > 0]                                                # 去除為0資料
        x = x.flatten()                                                         # 攤平數據
        
        flirframe_distribution_Left = normalObject.copy()
        flirframe_distribution_right = normalObject.copy()


        flirframe_distribution_Left[flirHot < conf_intveral[0]] = 0         
        flirframe_distribution_right[flirHot < conf_intveral[1]] = 0    

        return flirframe_distribution_Left, flirframe_distribution_right


    def saveCmap(self, flirHot, flirMode, pltSavepath = None):           # 轉換色彩地圖後儲存
        savePath = os.path.join(saveImgpath, str('Ischemia_') + imgName)
        flirframe_distribution_Left, flirframe_distribution_right = self.flirframe_distribution(flimask, confidence = 0.6826)               # 畫出左右標準差的值(缺血與發炎)
        
        distribution_save = []
        if flirMode == 'Ischemia':                                      # 如果是缺血使用左邊標準差數據
            distribution_save = flirframe_distribution_Left
        elif flirMode == 'Infect':                                      # 如果是發炎使用右邊標準差數據
            distribution_save = flirframe_distribution_right

        
        if pltSavepath:                                                                             # 如果有儲存地址
            print("save at:"+ str(pltSavepath))
            plt.axis('off')                                                                             # 關閉邊框
            plt.imsave(pltSavepath, flirHot, cmap=cm.gnuplot2)                                      # 使用plt儲存轉換色彩地圖的影像
            plt.imsave(pltSavepath, distribution_save, cmap=cm.gnuplot2)                                      # 使用plt儲存轉換色彩地圖的影像
            plt.close('all')                                                                        # 不顯示影像
        else:
            plt.imshow(flirHot, cmap=cm.gnuplot2)                                                       # 顯示溫度影像
            plt.imshow(distribution_save, cmap=cm.gnuplot2)
            plt.show()


if __name__ == '__main__':
    imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片\Ischemia FLIR')   # 輸入路徑
    saveImgpath = r'splitImg\flir_background\0_Ischemia'
    palettes = [cm.gnuplot2]                        # 影像調色板

    flirSplit = flir_img_split(imgInputpath, palettes)

    for imgPath in flirSplit.getImglist():
        flirRGB, flirHot = flirSplit.separateNP(imgPath)
        flimask, normalObject, hotObject = flirSplit.makeMask(flirHot)
        imgName = os.path.split(imgPath)[-1]
        
        flirSplit.saveCmap(flirHot, flirMode = 'Ischemia', pltSavepath = None)

        # break
        

