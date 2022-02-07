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
# drawHist = False                                # 畫出直方圖
# drawMask = True                                # 劃出遮罩

# minTemp = maxTemp = None                        # 溫度上下限

class FlirPretreatment():
    def __init__(self, imgPath, savePath, palettes):
        self.imgPath = imgPath
        self.savePath = savePath
        self.palettes = palettes
        self.imgPathlist = self.getImglist()

    def getImglist(self):                                                           # 遍歷路徑
        imgPathlist = []
        for path, dir_list, file_list in self.imgPath: 
            for file_name in file_list:
                imgPathlist.append(os.path.join(str(path), str(file_name)))
        
        return imgPathlist

    def drawHist(self, flirHot, flirMean=0):                                          # 畫出溫度直線圖
        plt.hist(flirHot, 35, [15, 35])     # 繪製直線圖
        plt.xlim([15, 35])
        plt.ylim([0, 400])

        if flirMean != 0:                   # 畫出均線
            plt.vlines(flirMean, 2, 400, color="red")     
            # print('mean :', flirMean)
        plt.title("Thermal Distribution")
        plt.xlabel("Thermal")
        plt.ylabel("Value")

        plt.text(31.45, 381.5, "Mean : " + str(round(flirMean, 2)),                 # 放置文字
                fontsize=15,
                color="red",
                verticalalignment ='top', 
                horizontalalignment ='center',
                bbox ={'facecolor':'white', 
                        'pad':10}
        )
        plt.show()
        # plt.close('all')

    def drawHist_frame(self, flirHot, flirframe=0):                                   # 畫出溫度範圍直線圖
        plt.hist(flirHot, 35, [15, 35])     # 繪製直線圖
        plt.xlim([15, 35])
        plt.ylim([0, 400])

        if flirframe != 0:
            plt.vlines(flirframe, 2, 400, color="blue")     
            plt.title("Thermal Distribution")
            plt.xlabel("Thermal")
            plt.ylabel("Value")

            plt.text(31.45, 381.5, "frame: " + str(round(flirframe, 2)),                 # 放置文字
                    fontsize=15,
                    color="blue",
                    verticalalignment ='top', 
                    horizontalalignment ='center',
                    bbox ={'facecolor':'white', 
                            'pad':10}
            )
        plt.show()
        # plt.close('all')

    def drawHist_Distribution(self, flimask, flirframe, confidence, normal, pltSavepath, drawLine):        # 畫出溫度範圍直線圖
        x = flimask[flimask > 0]                        # 去除為0資料
        x = x.flatten()                                 # 攤平數據
        mean, std = x.mean(), x.std(ddof=1)             # 計算均值與標準差
        dataRange = x.mean()
        conf_intveral = stats.norm.interval(confidence, loc=mean, scale=std)        # 取得信心區間 
        # print(mean, std)
        print("與平均溫差"+str(flirframe) + "度", dataRange + flirframe)
        print(str(confidence*100) + "%信賴區間", conf_intveral)

        fig, ax1 = plt.subplots()
        plt.title("Thermal Distribution")
        plt.xlabel("Thermal")
        plt.xlim([15, 35])
        # ax2 = ax1.twinx()
        if normal == False:                                                         # 是否使用 normal 效果
            ax1.set_ylabel("Value")
            if drawLine:
                l1 = ax1.vlines(mean, 0, 7000, linestyles ="-", color="red")
                l2 = ax1.vlines(dataRange + flirframe, 0, 7000, linestyles ="dotted", color="orange")
                l3 = ax1.vlines(conf_intveral, 0, 7000, linestyles ="-.", color="green")
            ax1 = sns.distplot(x, bins = 35, norm_hist=False, kde=False) 
        else:
            ax1.set_ylabel("Distribution")
            if drawLine:
                l1 = ax1.vlines(mean, 0, 0.5, linestyles ="-", color="red")
                l2 = ax1.vlines(dataRange + flirframe, 0, 0.5, linestyles ="dotted", color="orange")
                l3 = ax1.vlines(conf_intveral, 0, 0.5, linestyles ="-.", color="green")
            ax1 = sns.distplot(x, bins = 35, norm_hist=False, hist=True, kde=False, fit=stats.norm)                       # 使用SNS繪製，強制擬合常態分佈
        if drawLine:
            plt.legend(handles=[l1,l2,l3],labels=['mean','mean+'+str(flirframe),str(confidence*100)+'%'],loc='upper right')
    
        fig.tight_layout()
        if pltSavepath:
            print("save at:"+ str(pltSavepath))
            fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        plt.show()
        return conf_intveral

    def drawMask(self, flirRGB, flirHot, flimask, normalObject, pltSavepath):       # 畫出患者範圍
        flirRGB = cv2.resize(flirRGB, (int(flirHot.shape[1]), int(flirHot.shape[0])))

        fig = plt.figure()
        subplot1=fig.add_subplot(1, 4, 1)       
        subplot1.imshow(flirRGB)                            # 顯示 RGB影像
        subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(1, 4, 2)
        subplot2.imshow(flirHot, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")                    # 溫度影像

        subplot3=fig.add_subplot(1, 4, 3)
        subplot3.imshow(flimask)
        subplot3.set_title("Thresh Mask")                   # 遮罩影像

        subplot4=fig.add_subplot(1, 4, 4)
        subplot4.imshow(normalObject, cmap=cm.gnuplot2)
        subplot4.set_title("Flir Matting")

        # figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
        # fig.suptitle("")
        fig.tight_layout()

        if pltSavepath:
            fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        plt.show()

    def drawFrame(self, flirRGB, flirHot, normalObject, thermalRange, pltSavepath):      # 圈出溫差範圍

        # flirHot[flimask < 255] = 0
        flirframe = normalObject.copy()
        flirframe[flirHot < thermalRange] = 0         


        fig = plt.figure()
        subplot1=fig.add_subplot(1, 3, 1)
        subplot1.imshow(flirRGB)
        subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(1, 3, 2)
        subplot2.imshow(normalObject, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")

        subplot3=fig.add_subplot(1, 3, 3)
        subplot3.imshow(flirframe, cmap=cm.gnuplot2)
        subplot3.set_title("Flir Frame : "+ str(round(thermalRange, 2)))

        # subplot4=fig.add_subplot(1, 4, 4)
        # subplot4.imshow(normalObject-flirframe, cmap=cm.gnuplot2)
        # subplot4.set_title("Flir Frame XOR")


        figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MEAN Thermal :"+ str(round(np.mean(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
        fig.suptitle(figTitle)
        fig.tight_layout()

        if pltSavepath:
            print("save at:"+ str(pltSavepath))
            fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        plt.show()

    def drawAllframe(self, flirRGB, flirHot, normalObject, flimask, thermalRange, flirframe, confidence, pltSavepath):      # 圈出溫差範圍

        # flirHot[flimask < 255] = 0
        x = flimask[flimask > 0]                                                # 去除為0資料
        x = x.flatten()                                                         # 攤平數據
        dataRange = x.mean()                                                    # 設定標準值
        
        flirframe_4 = normalObject.copy()
        flirframe_distribution_Left = normalObject.copy()
        flirframe_distribution_right = normalObject.copy()

        flirframe_4[flirHot < (dataRange + flirframe)] = 0                      # 要畫出的範圍

        flirframe_distribution_Left[flirHot < thermalRange[0]] = 0         
        flirframe_distribution_right[flirHot < thermalRange[1]] = 0         

        fig = plt.figure()
        subplot1=fig.add_subplot(2, 3, 1)
        subplot1.imshow(flirRGB)
        subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(2, 3, 2)
        subplot2.imshow(flirHot, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")

        subplot3=fig.add_subplot(2, 3, 3)
        subplot3.imshow(normalObject, cmap=cm.gnuplot2)
        subplot3.set_title("Remove Background")

        subplot4=fig.add_subplot(2, 3, 4)
        subplot4.imshow(flirframe_4, cmap=cm.gnuplot2)
        subplot4.set_title(str(flirframe) +" frame")

        subplot5=fig.add_subplot(2, 3, 5)
        subplot5.imshow(flirframe_distribution_Left, cmap=cm.gnuplot2)
        subplot5.set_title("Distribution Left")

        subplot6=fig.add_subplot(2, 3, 6)
        subplot6.imshow(flirframe_distribution_right, cmap=cm.gnuplot2)
        subplot6.set_title("Distribution Right")

        figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MEAN Thermal :"+ str(round(np.mean(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
        fig.suptitle(figTitle)
        fig.tight_layout()

        if pltSavepath:
            pltSavepath = savePath.split(".")[0] + "_temp_" + str(flirframe) +"_distri_"+str(confidence*100)+".jpg"
            print("save at:"+ str(pltSavepath))
            fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        plt.show()

    def draw3D(self, flirObject, hotObject, flirHot, pltSavepath):                  # 畫出地形圖
        x_size = flirObject.shape[1]
        y_size =  flirObject.shape[0]
        x = np.linspace(0, x_size-1, x_size)
        y = np.linspace(0, y_size-1, y_size)

        # 將原始資料變成網格資料
        X,Y = np.meshgrid(x,y)
        # Z = flirHot

        # 填充顏色
        plt.contourf(X, Y, flirObject, 8, alpha = 0.75, cmap = plt.cm.gnuplot2)
        # add contour lines

        # C = plt.contour(X,Y,hotObject,8,color='black',lw=0.5)
        C = plt.contour(X,Y,hotObject,8)
        # 顯示各等高線的資料標籤cmap=plt.cm.hot

        plt.clabel(C,inline=True,fontsize=10)

        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)   #生成一個3d物件
        fig.add_axes(ax)

        ax.set_xlabel('image_X')
        ax.set_ylabel('image_Y')
        ax.set_zlabel('Thermal')
        ax.plot_surface(X, Y, flirHot, rstride = 1, cstride = 1, cmap = plt.cm.gnuplot2)                  # 生成一個曲面
        # fig.tight_layout()
        if pltSavepath:
            fig.savefig(pltSavepath, dpi = 1000, bbox_inches = 'tight')
            plt.close('all')
        plt.show()

    def separateNP(self, imgPath):                                                  # 分離原始圖像與溫度影像
        flir = flirimageextractor.FlirImageExtractor(palettes=self.palettes)                       # 熱影像轉換套件
        flir.process_image(imgPath)       
        flirRGB = flir.extract_embedded_image()                                                     # 輸出 RGB
        flirHot = flir.get_thermal_np()                                                             # 輸出熱影像資訊
        
        return flirRGB, flirHot
    
    def makeMask(self, flirHot, autoNormal):                                        # 圈出溫差 N度內範圍 
        flirMean = flirHot.mean()                                                                 # 計算整張熱影像平均                 
        
        flimask = flirHot.copy()        
        flimask[flirHot < flirMean] = 0                                                             # 產生遮罩

        normalObject = autoNormal.copy()                                                            # 二質化
        normalObject[flimask < flirMean] = 0

        hotObject = flirHot.copy()
        hotObject[flimask < 255] = 0
        return flimask, normalObject, hotObject

    def main(self):
        for imgPath in self.imgPathlist:
            imgName = os.path.split(imgPath)[-1]
            if imgName.split(".")[-1] == "jpg":
                savePath = os.path.join(self.savePath, imgName)
                
                flirRGB, flirHot = self.separateNP(imgPath)                                             # 分離原始圖像與溫度影像

                autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
                flimask, normalObject, hotObject = self.makeMask(flirHot, autoNormal)                   # 建立遮罩影像
                flirMean = flirHot.mean() 

                print("===========================")
                print(imgPath)
                print("MAX Thermal :", np.amax(flirHot))
                print("MIN Thermal :", np.amin(flirHot))

                # self.drawHist(flirHot, flirMean)
                # self.drawHist(flirHot, flirMean)
                temperDiffer = -1.5
                confiDence = 0.6826
                self.drawHist_frame(flimask)
                # conf_intveral = self.drawHist_Distribution(flimask, flirframe = temperDiffer, confidence = confiDence, normal = True, pltSavepath=None, drawLine = True)   # 畫出直線圖與分佈曲線

                # # flimask[flimask < np.amax(flimask) - 4] = 0
                # self.drawHist_frame(flimask, np.amax(flimask) - 4)                # 查看分佈

                # # self.drawMask(flirRGB, flirHot, flimask, normalObject, pltSavepath=None)
                # self.drawFrame(flirRGB, flirHot, normalObject, thermalRange = np.amax(flirHot) - 4, pltSavepath=None)   # 圈出溫差 N度內範圍 
                # self.drawFrame(flirRGB, flirHot, normalObject, thermalRange = conf_intveral[0], pltSavepath=None)       # 圈出溫差 N度內範圍 
                # self.drawFrame(flirRGB, flirHot, normalObject, thermalRange = conf_intveral[1], pltSavepath=None)       # 圈出溫差 N度內範圍 

                
                self.drawAllframe(flirRGB, flirHot, normalObject, flimask, thermalRange = conf_intveral, flirframe = temperDiffer, confidence = confiDence, pltSavepath=None)
                # # self.draw3D(normalObject, hotObject, flirHot, pltSavepath=None)
                break


if __name__ == '__main__':
    palettes = [cm.gnuplot2]                        # 影像調色板
    imgPath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片\Ischemia FLIR')   # 輸入路徑
    # imgPath = os.walk(r'sample\\all_information')   # 輸入路徑
    savePath = r'sample\\frame_save\\感染前期_標準差與溫差4度檢視'

    flir = FlirPretreatment(imgPath, savePath, palettes)
    flir.main()



# %%
