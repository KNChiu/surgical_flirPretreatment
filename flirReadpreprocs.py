#%%
import flirimageextractor
from matplotlib import cm
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2

minTemp = maxTemp = None                        # 溫度上下限
palettes = [cm.gnuplot2]                        # 影像調色板

imgPath = os.walk(r'sample\\all_information')   # 輸入路徑

for path, dir_list, file_list in imgPath:  
    for file_name in file_list:                           
        hotPath = os.path.join(str(path), str(file_name))                                           # 熱影像路徑
        pltSavepath = os.path.join(str('sample\\plt_save'), str(file_name.split('.')[0]+".jpg"))    # 輸出路徑


        flir = flirimageextractor.FlirImageExtractor(palettes=palettes)                             # 熱影像轉換套件
        # print(hotPath)
        flir.process_image(hotPath)                                         
        flirRGB = flir.extract_embedded_image()                                                     # 輸出 RGB
        flirHot = flir.get_thermal_np()                                                             # 輸出 1/2 大小熱影像資訊


        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))           # 標準化到 0~1 之間
    

        print("MAX Thermal :", np.amax(flirHot))
        print("MIN Thermal :", np.amin(flirHot))


        # print("MAX Normal :", np.amax(thermal_normalized))
        # print("Min Normal :", np.amin(thermal_normalized))


        # print("MAX AutoNormal :", np.amax(autoNormal))
        # print("Min AutoNormal :", np.amin(autoNormal))

        flirMean = flirHot.mean()                                                                 # 計算整張熱影像平均
        ret, flimask = cv2.threshold(flirHot, flirMean, 255, cv2.THRESH_BINARY)                   # 產生遮罩

        autoNormal[flimask < 255] = 0                                                             # 二質化

        fig = plt.figure()
        subplot1=fig.add_subplot(1, 4, 1)
        subplot1.imshow(flirRGB)
        subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(1, 4, 2)
        subplot2.imshow(autoNormal, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")

        subplot3=fig.add_subplot(1, 4, 3)
        subplot3.imshow(flimask)
        subplot3.set_title("Thresh Mask")

        subplot4=fig.add_subplot(1, 4, 4)
        subplot4.imshow(autoNormal, cmap=cm.gnuplot2)
        subplot4.set_title("Flir Matting")

        # fig.suptitle("Flir Image")
        fig.tight_layout()
        fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
        plt.close('all')


