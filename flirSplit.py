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

    def getImglist(self):
        """                           
        自訂函數 : 取得每個影像路徑
        """

        imgPathlist = []
        for path, dir_list, file_list in self.imgPath:                               
            for file_name in file_list:
                if file_name.split(".")[-1] == "jpg":                                               # 只保留副檔名為*.jpg的影像
                    imgPathlist.append(os.path.join(str(path), str(file_name)))
                
        return imgPathlist

    def separateNP(self, imgPath):     
        """                           
        自訂函數 : 分離原始圖像與溫度影像
        """

        flir = flirimageextractor.FlirImageExtractor(palettes=self.palettes)                        # 熱影像轉換套件
        flir.process_image(imgPath)       
        flirRGB = flir.extract_embedded_image()                                                     # 輸出 RGB
        flirHot = flir.get_thermal_np()                                                             # 輸出熱影像資訊
        
        return flirRGB, flirHot
    
    def drawMeanhist(self, flirHot, imgName):    
        """                           
        自訂函數 : 畫出溫度分佈與背景均值位置
        """            
        # 原始輸入數據，直線圖用
        flirFlatten = flirHot.flatten()             # 攤平數據
        flirMean = flirFlatten.mean()               # 計算均值
        

        # 去除背景(均值)後，直線圖用
        flirHistremove = flirFlatten.copy()
        flirHistremove[flirHistremove < flirMean] = 0

        flirBoundary = flirHot.copy()                   # 患部與邊緣邊界
        flirBoundary[flirHot <= (flirMean - 0.5)] = 0   # 均值剛好為患部與背景的邊緣
        flirBoundary[flirHot >= (flirMean + 0.5)] = 0

        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
        normalObject = autoNormal.copy()                                                        
        normalObject[flimask < flirMean] = 0                                                    # 去背景


        fig, ax1 = plt.subplots()
        plt.title("Thermal Distribution")
        plt.xlabel("Thermal")
        plt.xlim([15, 35])

        ax1.set_ylabel("accumulation")
        l1 = ax1.vlines(flirMean, 0, 70000, linestyles ="-", color="red")                       # 畫出均值位置
        ax1 = sns.distplot(flirHistremove, bins = 35, norm_hist=False, kde=False) 
        # plt.legend(handles=[l1], labels=['Thermal mean'], loc='upper right')                    # 圖例
        fig.tight_layout()

        fig1 = plt.figure()
        subplot1=fig1.add_subplot(1, 3, 1)
        subplot1.imshow(flirHot, cmap=cm.gnuplot2)              # 溫度影像
        subplot1.set_title("Flir Image")

        subplot2=fig1.add_subplot(1, 3, 2)
        subplot2.imshow(flirBoundary, cmap=cm.gnuplot2)         # 患部與背景邊界
        subplot2.set_title("Flir Boundary")

        subplot3=fig1.add_subplot(1, 3, 3)
        subplot3.imshow(normalObject, cmap=cm.gnuplot2)         # 去除背景後影像
        subplot3.set_title("Remove Background")
        fig1.tight_layout()

        # figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MEAN Thermal :"+ str(round(np.mean(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
        # fig.suptitle(figTitle)
        
        pathNoextension = imgName.split('.')[0]

        fig1_pltSavepath = fig_pltSavepath = None
        # fig1_pltSavepath = r'結果存圖\論文\熱影像_邊界_去背比較' + '\\' + pathNoextension + "_fire_boundary_remove.jpg"     # 患部影像
        fig_pltSavepath = r'結果存圖\論文\溫度分佈狀況\去除背景' + '\\' + pathNoextension + "_fire_background_remove_hist.jpg"     # 患部影像


        if fig1_pltSavepath:
            print("save at:"+ str(fig1_pltSavepath))
            fig1.savefig(fig1_pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')

        if fig_pltSavepath:
            print("save at:"+ str(fig_pltSavepath))
            fig.savefig(fig_pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')

        plt.show()
        # return conf_intveral
    
    def makeMask(self, flirHot):                                 
        """                           
        自訂函數 : 圈出溫差 N度內範圍 
        """

        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
        flirMean = flirHot.mean()                                                                 # 計算整張熱影像平均                 
        
        flimask = flirHot.copy()        
        flimask[flirHot < flirMean] = 0                                                             # 產生遮罩

        normalObject = autoNormal.copy()                                                            # 二質化
        normalObject[flimask < flirMean] = 0

        hotObject = flirHot.copy()
        hotObject[flimask < 255] = 0

        return flimask, normalObject, hotObject

    def drawMask(self, flirRGB, flirHot, flimask, normalObject, pltSavepath):       
        """                           
        自訂函數 : 畫出患者範圍
        """
    
        flirRGB = cv2.resize(flirRGB, (int(flirHot.shape[1]), int(flirHot.shape[0])))

        fig = plt.figure()
        subplot1=fig.add_subplot(1, 3, 1)       
        subplot1.imshow(flirRGB)                            # 顯示 RGB影像
        subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(1, 3, 2)
        subplot2.imshow(flirHot, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")                    # 溫度影像

        # subplot3=fig.add_subplot(1, 4, 3)
        # subplot3.imshow(flimask)
        # subplot3.set_title("Thresh Mask")                   # 遮罩影像

        subplot3=fig.add_subplot(1, 3, 3)
        subplot3.imshow(normalObject, cmap=cm.gnuplot2)
        subplot3.set_title("Image Matting")

        # figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
        # fig.suptitle("")
        fig.tight_layout()

        if pltSavepath:
            fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        plt.show()


    def flirframe_distribution(self, flimask, confidence):
        """                           
        自訂函數 : 畫出左右標準差的值
        """

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


    def saveCmap(self, flirHot, flirMode, pltSavepath = None):           
        """                           
        自訂函數 : 轉換色彩地圖後儲存
        """
        
        flirframe_distribution_Left, flirframe_distribution_right = self.flirframe_distribution(flimask, confidence = 0.6826)               # 畫出左右標準差的值(缺血與發炎)
        
        distribution_save = []
        # print("flirMode :", flirMode)
        # if flirMode == 'Ischemia':                                      # 如果是缺血使用左邊標準差數據
        #     distribution_save = flirframe_distribution_Left
        # elif flirMode == 'Infect':                                      # 如果是發炎使用右邊標準差數據
            # distribution_save = flirframe_distribution_right

        distribution_save = flirframe_distribution_Left

        if pltSavepath:                                                                # 如果有儲存地址
            pathNoextension = pltSavepath.split('.')[0]
            flirPath = pathNoextension + '_flir' + '.jpg'
            distPath = pathNoextension + '_dist' + '.jpg'
            print("save at:"+ str(flirPath) + ', ' + str(distPath))
            plt.axis('off')                                                                             # 關閉邊框
            plt.imsave(flirPath, flirHot, cmap=cm.gnuplot2)                                      # 使用plt儲存轉換色彩地圖的影像
            # plt.imsave(distPath, distribution_save, cmap=cm.gnuplot2)                                      # 使用plt儲存轉換色彩地圖的影像
            plt.close('all')                                                                        # 不顯示影像
        else:
            plt.imshow(flirHot, cmap=cm.gnuplot2)                                                       # 顯示溫度影像
            plt.show()
            plt.imshow(distribution_save, cmap=cm.gnuplot2)
            plt.show()


if __name__ == '__main__':
    imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片')   # 輸入路徑
    saveImgpath = r'結果存圖\論文\原始影像_熱影像_去背'
    palettes = [cm.gnuplot2]                        # 影像調色板

    flirSplit = flir_img_split(imgInputpath, palettes)

    for imgPath in flirSplit.getImglist():
        # print(imgPath)
        imgName = os.path.split(imgPath)[-1]
        savePath = os.path.join(saveImgpath, imgName)

        flirRGB, flirHot = flirSplit.separateNP(imgPath)
        flimask, normalObject, hotObject = flirSplit.makeMask(flirHot)
        flirSplit.drawMask(flirRGB, flirHot, flimask, normalObject, pltSavepath = savePath)

        

        # flirSplit.drawMeanhist(flirHot, imgName)         # 畫出背景與患部溫度分佈圖

        # flirSplit.saveCmap(normalObject, flirMode = 'Infect', pltSavepath = None)

        # break
        

