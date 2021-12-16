#%%

import os
from matplotlib import cm
import numpy as np
from flirTool import FlirPretreatment
import matplotlib.pyplot as plt


def drawPLT(img, title, pltSavepath):
    fig = plt.figure()
    # for i in range(len(img)):
    subplot1=fig.add_subplot(1, 1, 1)
    subplot1.imshow(img, cmap=cm.gnuplot2)
    subplot1.set_title(str(title))

        # subplot2=fig.add_subplot(1, len(img), 2)
        # subplot2.imshow(img[0], cmap=cm.gnuplot2)
        # subplot2.set_title("Flir image")

        # subplot3=fig.add_subplot(1, len(img), 3)
        # subplot3.imshow(img[0])
        # subplot3.set_title("Thresh Mask")

        # subplot4=fig.add_subplot(1, len(img), 4)
        # subplot4.imshow(img[0], cmap=cm.gnuplot2)
        # subplot4.set_title("Flir Matting")

    # figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
    # fig.suptitle("")
    fig.tight_layout()

    if pltSavepath:
        fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
        plt.close('all')
    plt.show()


if __name__ == '__main__':
    palettes = [cm.gnuplot2]                        # 影像調色板
    imgPath = os.walk(r'sample\all_information')   # 輸入路徑
    # imgPath = os.walk(r'sample\\all_information')   # 輸入路徑
    savePath = r'sample\\frame_save\\等溫線圖'

    flir = FlirPretreatment(imgPath, savePath, palettes)
    for imgPath in flir.imgPathlist:
        imgName = os.path.split(imgPath)[-1]
        savePath = os.path.join(flir.savePath, imgName)
        
        flirRGB, flirHot = flir.separateNP(imgPath)                                             # 分離原始圖像與溫度影像

        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
        flimask, normalObject, hotObject = flir.makeMask(flirHot, autoNormal)                   # 建立遮罩影像
        flirMean = flirHot.mean() 

        print("===========================")
        print(imgPath)
        print("MAX Thermal :", np.amax(flirHot))
        print("MIN Thermal :", np.amin(flirHot))
        
        # img = []
        # title = []
        img = hotObject
        title = "mask"
        break
        # drawPLT(img, title, pltSavepath=None)


        # flir.drawHist(flirHot, flirMean)
        # flir.drawHist(hotObject, flirMean)

        # flir.drawMask(flirRGB, flirHot, flimask, normalObject, pltSavepath=None)
        # self.drawFrame(flirRGB, flirHot, normalObject, thermalRange = 4, pltSavepath=savePath)       # 圈出溫差 N度內範圍 
        # self.draw3D(normalObject, hotObject, flirHot, pltSavepath=None)

        