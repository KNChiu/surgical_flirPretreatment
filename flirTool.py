#%%
import flirimageextractor
from matplotlib import cm
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2
from mpl_toolkits.mplot3d import Axes3D

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

    def drawHist(self, flirHot, flirMean):                                          # 畫出溫度直線圖
        print('mean :', flirMean)
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

        plt.hist(flirHot, 35, [15, 35])     # 繪製直線圖
        plt.xlim([15, 35])
        plt.ylim([0, 400])
        plt.vlines(flirMean, 2, 400, color="red")                          
        plt.show()
        # plt.close('all')

    def drawMask(self, flirRGB, flirHot, flimask, normalObject, pltSavepath):       # 畫出患者範圍
        fig = plt.figure()
        subplot1=fig.add_subplot(1, 4, 1)
        subplot1.imshow(flirRGB)
        subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(1, 4, 2)
        subplot2.imshow(flirHot, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")

        subplot3=fig.add_subplot(1, 4, 3)
        subplot3.imshow(flimask)
        subplot3.set_title("Thresh Mask")

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
        flirframe[flirHot < np.amax(flirHot) - thermalRange] = 0         


        fig = plt.figure()
        subplot1=fig.add_subplot(1, 3, 1)
        subplot1.imshow(flirRGB)
        subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(1, 3, 2)
        subplot2.imshow(normalObject, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")

        subplot3=fig.add_subplot(1, 3, 3)
        subplot3.imshow(flirframe, cmap=cm.gnuplot2)
        subplot3.set_title("Flir Frame - "+ str(thermalRange))

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
        flirHot = flir.get_thermal_np()                                                             # 輸出 1/2 大小熱影像資訊
        
        return flirRGB, flirHot
    
    def makeMask(self, flirHot, autoNormal):                                        # 圈出溫差 N度內範圍 
        flirMean = flirHot.mean()                                                                 # 計算整張熱影像平均                 
        
        flimask = flirHot.copy()        
        flimask[flirHot < flirMean] = 0                                                             # 產生遮罩

        normalObject = autoNormal.copy()                                                            # 二質化
        normalObject[flimask < 255] = 0

        hotObject = flirHot.copy()
        hotObject[flimask < 255] = 0
        return flimask, normalObject, hotObject

    def main(self):
        for imgPath in self.imgPathlist:
            imgName = os.path.split(imgPath)[-1]
            savePath = os.path.join(self.savePath, imgName)
            
            flirRGB, flirHot = self.separateNP(imgPath)                                             # 分離原始圖像與溫度影像

            autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
            flimask, normalObject, hotObject = self.makeMask(flirHot, autoNormal)                   # 建立遮罩影像
            flirMean = flirHot.mean() 

            print("===========================")
            print(imgPath)
            print("MAX Thermal :", np.amax(flirHot))
            print("MIN Thermal :", np.amin(flirHot))

            self.drawHist(flirHot, flirMean)
            self.drawHist(flimask, flirMean)
            # self.drawMask(flirRGB, flirHot, flimask, normalObject, pltSavepath=None)
            # self.drawFrame(flirRGB, flirHot, normalObject, thermalRange = 4, pltSavepath=savePath)       # 圈出溫差 N度內範圍 
            # self.draw3D(normalObject, hotObject, flirHot, pltSavepath=None)
            break


if __name__ == '__main__':
    palettes = [cm.gnuplot2]                        # 影像調色板
    imgPath = os.walk(r'sample\all_information')   # 輸入路徑
    # imgPath = os.walk(r'sample\\all_information')   # 輸入路徑
    savePath = r'sample\\frame_save\\等溫線圖'

    flir = FlirPretreatment(imgPath, savePath, palettes)
    flir.main()

