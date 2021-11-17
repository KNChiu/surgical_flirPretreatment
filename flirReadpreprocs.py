#%%
import flirimageextractor
from matplotlib import cm
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

drawHist = False                                # 畫出直方圖
drawMask = True                                # 劃出遮罩

minTemp = maxTemp = None                        # 溫度上下限
palettes = [cm.gnuplot2]                        # 影像調色板

imgPath = os.walk(r'sample\\all_information')   # 輸入路徑
savePath = r'sample\\plt_save'



def draw3D(flirObject, hotObject):
    x_size = flirObject.shape[1]
    y_size =  flirObject.shape[0]
    x = np.linspace(0, x_size-1, x_size)
    y = np.linspace(0, y_size-1, y_size)

    # 將原始資料變成網格資料
    X,Y = np.meshgrid(x,y)
    # Z = flirHot

    # 填充顏色
    plt.contourf(X,Y,flirObject,8,alpha=0.75,cmap=plt.cm.gnuplot2)
    # add contour lines

    C = plt.contour(X,Y,hotObject,8,color='black',lw=0.5)
    # 顯示各等高線的資料標籤cmap=plt.cm.hot

    plt.clabel(C,inline=True,fontsize=10)


    fig = plt.figure()
    ax = Axes3D(fig)     #生成一個3d物件

    ax.set_xlabel('image_X')
    ax.set_ylabel('image_Y')
    ax.set_zlabel('Thermal')
    ax.plot_surface(X, Y, flirHot,rstride=1,cstride=1,cmap=plt.cm.gnuplot2)                         # 生成一個曲面

    fig.tight_layout()

    plt.show()

for path, dir_list, file_list in imgPath:  
    for file_name in file_list:                           
        hotPath = os.path.join(str(path), str(file_name))                                           # 熱影像路徑
        pltSavepath = os.path.join(str(savePath), str(file_name.split('.')[0]+".jpg"))    # 輸出路徑


        flir = flirimageextractor.FlirImageExtractor(palettes=palettes)                             # 熱影像轉換套件
        # print(hotPath)
        flir.process_image(hotPath)       
        flirRGB = flir.extract_embedded_image()                                                     # 輸出 RGB
        flirHot = flir.get_thermal_np()                                                             # 輸出 1/2 大小熱影像資訊


        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))           # 標準化到 0~1 之間
    

        print("MAX Thermal :", np.amax(flirHot))
        print("MIN Thermal :", np.amin(flirHot))

        
        flirMean = flirHot.mean()                                                                 # 計算整張熱影像平均
        ret, flimask = cv2.threshold(flirHot, flirMean, 255, cv2.THRESH_BINARY)                   # 產生遮罩

        normalObject = autoNormal.copy()                                                            # 二質化
        normalObject[flimask < 255] = 0

        hotObject = flirHot.copy()
        hotObject[flimask < 255] = 0


        if drawHist:
            print('mean :', flirMean)
            plt.title("Thermal Distribution")
            plt.xlabel("Thermal")
            plt.ylabel("Value")

            plt.text(31.45, 477.5, "Mean : " + str(round(flirMean, 2)),
                    fontsize=15,
                    color="red",
                    verticalalignment ='top', 
                    horizontalalignment ='center',
                    bbox ={'facecolor':'white', 
                            'pad':10}
            )

            plt.hist(flirHot)
            plt.xlim([15, 35])
            plt.ylim([0, 500])
            plt.vlines(flirMean, 1, 400, color="red")                          
            plt.show()
            plt.close('all')

        if drawMask:
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

            # fig.suptitle("Flir Image")
            fig.tight_layout()
            fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')
            plt.show()
        
        # draw3D(normalObject, hotObject)


