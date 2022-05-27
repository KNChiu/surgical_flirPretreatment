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
from PIL import Image

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
   
    def drawHist(self, flirHot, flimask, vline, labelName, savePath=None):
        '''自訂函數 : 繪製直方圖'''
        
        fig, ax1 = plt.subplots()
        plt.title("Thermal Distribution")
        plt.xlabel("Thermal")
        plt.xlim([15, 35])

        flirHot[flimask == 0] = 0

        ax1.set_ylabel("accumulation")
        ax1 = sns.distplot(flirHot, bins = 70, norm_hist=False, kde=False)                 # 畫出直方圖
        l1 = ax1.vlines(vline, 0, 25000, linestyles ="-", color="red")                  # 畫出均值位置
        plt.legend(handles=[l1], labels=[str(labelName)], loc='upper right')               # 圖例
        fig.tight_layout()

        fig_pltSavepath = None

        pathNoextension = savePath.split('.')[0]
        fig_pltSavepath = pathNoextension + "_hist_" + str(labelName) + ".jpg"       # 患部直線圖

        if fig_pltSavepath:
            print("save at:"+ str(fig_pltSavepath))
            fig.savefig(fig_pltSavepath, dpi=1000, bbox_inches='tight')
        else:
            plt.show()

        plt.close('all')

    def draw_MeanMin_hist(self, flirHot, imgName, outputImg, savePath):    
        """                           
        自訂函數 : 畫出(均值與區域最低)溫度分佈與背景均值與累績最低值位置
        """            
        # 原始輸入數據，直線圖用
        flirFlatten = flirHot.flatten()             # 攤平數據
        flirMean = flirFlatten.mean()               # 計算均值

        # 計算區域最低
        localrange = np.array(plt.hist(flirFlatten, bins = 70)[0:2], dtype=object)

        ## localrange[0] : 累積計數值
        ## localrange[1] : 對應溫度值


        offset = 8
        Meanindex = (np.abs(localrange[1]-flirMean)).argmin()                           # 找出最接近平均值在陣列中的位置 
        Minindex = localrange[0][Meanindex - offset : Meanindex + offset].argmin()      # 找出 Meanindex 正負 offst 範圍內累積計數值最低的位置 
        localMin = localrange[1][Meanindex - offset + Minindex]                         # 回傳範圍內累積數量最低的對應溫度值

        # savePath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\範例圖像\輸出影像'
        
        def draw_Mean_Min(flirMean, labelName):
            
            # 去除背景(均值)後，直線圖用
            flirHistremove = flirFlatten.copy()
            flirHistremove[flirHistremove < flirMean] = 0

            flirBoundary = flirHot.copy()                   # 遮罩
            # flirBoundary[flirHot <= (flirMean - 0.5)] = 0   # 均值剛好為患部與背景的邊緣
            # flirBoundary[flirHot >= (flirMean + 0.5)] = 0
            flirBoundary[flirHot >= flirMean] = 255
            flirBoundary[flirHot < flirMean] = 0


            autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
            normalObject = autoNormal.copy()    
            
            flimask = flirHot.copy()        
            flimask[flirHot < localMin] = 0                                                         # 產生遮罩                                                    
            normalObject[flimask < flirMean] = 0                                                    # 去背景


            # fig, ax1 = plt.subplots()
            # plt.title("Thermal Distribution")
            # plt.xlabel("Thermal")
            # plt.xlim([15, 35])

            # ax1.set_ylabel("accumulation")
            # ax1 = sns.distplot(flirHot, bins = 70, norm_hist=False, kde=False)                 # 畫出直方圖
            # l1 = ax1.vlines(flirMean, 0, 25000, linestyles ="-", color="red")                       # 畫出均值位置
            # plt.legend(handles=[l1], labels=[str(labelName)], loc='upper right')                    # 圖例
            # fig.tight_layout()

            fig1 = plt.figure()
            subplot1=fig1.add_subplot(1, 3, 1)
            subplot1.imshow(flirHot, cmap=cm.gnuplot2)              # 溫度影像
            subplot1.set_title("Flir Image")

            subplot2=fig1.add_subplot(1, 3, 2)
            subplot2.imshow(flirBoundary)                           # 遮罩
            subplot2.set_title("Thresh Mask")

            subplot3=fig1.add_subplot(1, 3, 3)
            subplot3.imshow(normalObject, cmap=cm.gnuplot2)         # 去除背景後影像
            subplot3.set_title("Remove Background")
            fig1.tight_layout()

            # figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MEAN Thermal :"+ str(round(np.mean(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
            # fig.suptitle(figTitle)
            if savePath:
                pathNoextension = savePath.split('.')[0]
                fig1_pltSavepath = pathNoextension + "_fire_remove_" + str(labelName) + ".jpg"     # 患部影像
                # fig_pltSavepath = savePath + '\\' + pathNoextension + "_fire_hist_" + str(labelName) + ".jpg"       # 患部直線圖

                print("save at:"+ str(fig1_pltSavepath))
                fig1.savefig(fig1_pltSavepath, dpi=1000, bbox_inches='tight')
                plt.close('all')

            # if fig_pltSavepath:
            #     print("save at:"+ str(fig_pltSavepath))
            #     fig.savefig(fig_pltSavepath, dpi=1000, bbox_inches='tight')
            #     plt.close('all')

            plt.show()
        plt.close('all')

            # return conf_intveral

        if outputImg :
            draw_Mean_Min(flirMean, 'Global Mean')
            draw_Mean_Min(localMin, 'Local Minimum')

        return localMin

    def fixMask(self, flimask, outputImg = False, pltSavepath = None):
        '''自訂函數 : 修復遮罩影像'''

        fig = plt.figure()
        subplot1=fig.add_subplot(1, 4, 1)       
        subplot1.imshow(flimask)                            
        subplot1.set_title("original")

        # 閉運算
        # kernel = np.ones((13,13), np.uint8)
        # flimask = cv2.dilate(flimask, kernel ,iterations=2)
        # flimask = cv2.erode(flimask, kernel ,iterations=2)

        # 尋找階層輪廓
        contours, hierarchy = cv2.findContours(flimask, cv2.RETR_CCOMP, 2)
        hierarchy = np.squeeze(hierarchy)       # (1, 6, 4) -> (6, 4)

        if hierarchy.ndim > 1:              # 如果有子類再進行修復
            for i in range(len(contours)):  # 找出父輪廓內的子輪廓填充
                if (hierarchy[i][3] != -1):
                    cv2.drawContours(flimask, contours, i, (255), -1)

        subplot2=fig.add_subplot(1, 4, 2)       
        subplot2.imshow(flimask)                            
        subplot2.set_title("Contours")

        # 開運算
        kernel = np.ones((13,13), np.uint8)
        flimask = cv2.erode(flimask, kernel ,iterations=2)
        flimask = cv2.dilate(flimask, kernel ,iterations=2)

        subplot3=fig.add_subplot(1, 4, 3)       
        subplot3.imshow(flimask)                            
        subplot3.set_title("Opening")


        # 使用連通域面積方式
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(flimask, connectivity=8)
        # num_labels : 連通域數量
        # labels : 對象素點標註
        # stats : 列出 x、y、width、height、面積
        for label in range(num_labels):         # 連通域數量
            if stats[label][4] < 20000:
                flimask[labels == label] = 0

        subplot4=fig.add_subplot(1, 4, 4)       
        subplot4.imshow(flimask)                            
        subplot4.set_title("Connected")

        fig.tight_layout()

        if pltSavepath:
            pathNoextension = pltSavepath.split('.')[0]
            flirPath = pathNoextension + '_fixMask' + '.jpg'
            fig.savefig(flirPath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        else:
            if outputImg:
                plt.show()
        plt.close('all')
        return flimask

    def makeMask(self, flirHot, localMin, fixmask = True, outputImg=False, pltSavepath = None):                                 
        """                           
        自訂函數 : 圈出溫差 N度內範圍 
        """
        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
        # flirMean = flirHot.mean()                                                                 # 計算整張熱影像平均                 

        flimask = flirHot.copy()        
        flimask[flirHot < localMin] = 0  
        flimask = flimask.astype(np.uint8)
        flimask[flimask > 0] = 255 
        
        if fixmask:
            flimask = self.fixMask(flimask, outputImg, pltSavepath)

        normalObject = autoNormal.copy()                                                            # 二質化
        normalObject[flimask == 0] = 0

        hotObject = flirHot.copy()
        hotObject[flimask == 0] = 0

        return flimask, normalObject, hotObject

    def drawMask(self, flirRGB, flirHot, flimask, normalObject, pltSavepath):       
        """                           
        自訂函數 : 畫出患者範圍
        """
    
        flirRGB = cv2.resize(flirRGB, (int(flirHot.shape[1]), int(flirHot.shape[0])))

        fig = plt.figure()
        # subplot1=fig.add_subplot(1, 4, 1)       
        # subplot1.imshow(flirRGB)                            # 顯示 RGB影像
        # subplot1.set_title("RGB image")

        subplot2=fig.add_subplot(1, 3, 1)
        subplot2.imshow(flirHot, cmap=cm.gnuplot2)
        subplot2.set_title("Flir image")                    # 溫度影像

        subplot3=fig.add_subplot(1, 3, 2)
        subplot3.imshow(flimask)
        subplot3.set_title("Thresh Mask")                   # 遮罩影像

        subplot3=fig.add_subplot(1, 3, 3)
        subplot3.imshow(normalObject, cmap=cm.gnuplot2)
        subplot3.set_title("Remove Background")

        # figTitle = "MAX Thermal :"+ str(round(np.amax(flirHot), 2))+ "  |  " + "MIN Thermal :"+ str(round(np.amin(flirHot), 2))
        # fig.suptitle("")
        fig.tight_layout()

        if pltSavepath:
            pathNoextension = pltSavepath.split('.')[0]
            flirPath = pathNoextension + '_fixFilr' + '.jpg'
            fig.savefig(flirPath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        else:
            plt.show()

    def flirframe_distribution(self, hotObject, confidence):
        """                           
        自訂函數 : 畫出左右標準差的值
        """

        x = hotObject[hotObject > 0]                        # 去除為0資料
        x = x.flatten()                                 # 攤平數據
        mean, std = x.mean(), x.std(ddof=1)             # 計算均值與標準差
        conf_intveral = stats.norm.interval(confidence, loc=mean, scale=std)        # 取得信心區間 
        print(str(confidence*100) + "%信賴區間", conf_intveral)
        
        flirframe_distribution_Left = normalObject.copy()
        flirframe_distribution_right = normalObject.copy()

        # flirframe_distribution_Left[hotObject > x.max()-4] = 0 
        flirframe_distribution_Left[hotObject > conf_intveral[0]] = 0

        flirframe_distribution_right[hotObject < conf_intveral[1]] = 0    

        return flirframe_distribution_Left, flirframe_distribution_right

    def saveCmap(self, flirHot, hotObject, flimask, pltSavepath = None):           
        """                           
        自訂函數 : 轉換色彩地圖後儲存
        """
        
        flirframe_distribution_Left, flirframe_distribution_right = self.flirframe_distribution(hotObject, confidence = 0.682)               # 畫出左右標準差的值(缺血與發炎)
        
        distribution_save = []
        # print("flirMode :", flirMode)
        # if flirMode == 'Ischemia':                                      # 如果是缺血使用左邊標準差數據
        #     distribution_save = flirframe_distribution_Left
        # elif flirMode == 'Infect':                                      # 如果是發炎使用右邊標準差數據
        #     distribution_save = flirframe_distribution_right

        fig = plt.figure()
        subplot1=fig.add_subplot(1, 3, 1)  
        subplot1.imshow(normalObject, cmap=cm.gnuplot2)                
        subplot1.set_title("normalObject")                                       # 顯示溫度影像

        subplot2=fig.add_subplot(1, 3, 2)  
        subplot2.imshow(flirframe_distribution_Left, cmap=cm.gnuplot2)                
        subplot2.set_title("Distribution Left")

        subplot3=fig.add_subplot(1, 3, 3)  
        subplot3.imshow(flirframe_distribution_right, cmap=cm.gnuplot2)                
        subplot3.set_title("Distribution right")
        fig.tight_layout()

        save_one = False

        # flirPath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\code\各項功能\結果存圖\論文\原始影像_熱影像_分隔去背比較\色彩地圖\\'+'2.jpg'
        # plt.axis('off')  
        # plt.imsave(flirPath, flirframe_distribution_Left, cmap=cm.gnuplot2)                

        # distribution_Left_image = plt.imread(flirPath)
        # image = distribution_Left_image.copy()
        # image[flimask==0] = [122, 195, 236]

        # flirPath1 = r'G:\我的雲端硬碟\Lab\Project\外科溫度\code\各項功能\結果存圖\論文\原始影像_熱影像_分隔去背比較\色彩地圖\\'+'5.jpg'
        # # plt.imshow(image)
        # plt.axis('off')  
        # plt.imsave(flirPath1, image) 
        # # plt.show()

        if pltSavepath:                                                                # 如果有儲存地址
            if save_one:
                pathNoextension = pltSavepath.split('.')[0]
                flirPath = pathNoextension + '_flir' + '.jpg'
                distPath = pathNoextension + '_dist' + '.jpg'
                print("save at:"+ str(flirPath) + ', ' + str(distPath))
                plt.axis('off')                                                                             # 關閉邊框
                plt.imsave(flirPath, normalObject, cmap=cm.gnuplot2)                                      # 使用plt儲存轉換色彩地圖的影像
                # plt.imsave(distPath, distribution_save, cmap=cm.gnuplot2)                                      # 使用plt儲存轉換色彩地圖的影像
            
            else:
                pathNoextension = pltSavepath.split('.')[0]
                Path = pathNoextension + '_distribution_compare' + '.jpg'
                fig.savefig(Path, dpi=1000, bbox_inches='tight')

            plt.close('all')                                                                        # 不顯示影像
        else:
            plt.show()

    def saveCmapimg(self, flirHot, flimask, pltSavepath = None, imgName = None):
        if pltSavepath:
            pathNoextension = pltSavepath.split('.')[0]
            flirPath = pathNoextension + '_cmap' + '.jpg'
            plt.axis('off')                                                                          # 關閉邊框
            plt.imsave(flirPath, flirHot, cmap=self.palettes[0])
            plt.close('all') 
        else:
            plt.show()

        flirHavebackground = np.array(Image.open(flirPath))
        hotObject = flirHavebackground.copy()
        hotObject[flimask == 0] = 0


        savePath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\訓練用數據集\removeBackground\\Ischemia FLIR'
        pathNoextension = savePath.split('.')[0]
        flirPath = pathNoextension + '/' + str(imgName.split('.')[0]) + '.jpg'
        plt.axis('off') 
        plt.imsave(flirPath, hotObject, cmap=self.palettes[0])
        plt.close('all') 

    def deviation_comparison(self, flirHot, hotObject, flimask):           
        """                           
        自訂函數 : 標準差比較、藍色背景
        """

        temperDiffer = 1.5  # 溫差
        confiDence = 0.6826  # 分佈機率

        distriPath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\code\各項功能\結果存圖\論文\原始影像_熱影像_分隔去背比較\色彩地圖\\'+'1.jpg'
        self.drawHist_Distribution(hotObject, flirframe = temperDiffer, confidence = confiDence, normal = True, pltSavepath = distriPath, drawLine = True)   # 畫出直線圖與分佈曲線
        flirframe_distribution_Left, flirframe_distribution_right = self.flirframe_distribution(hotObject, confidence = confiDence)               # 畫出左右標準差的值(缺血與發炎)
        
        # fig = plt.figure()
        # subplot1=fig.add_subplot(1, 3, 1)  
        # subplot1.imshow(normalObject, cmap=cm.gnuplot2)                
        # subplot1.set_title("normalObject")                                       # 顯示溫度影像

        # subplot2=fig.add_subplot(1, 3, 2)  
        # subplot2.imshow(flirframe_distribution_Left, cmap=cm.gnuplot2)                
        # subplot2.set_title("Distribution Left")

        # subplot3=fig.add_subplot(1, 3, 3)  
        # subplot3.imshow(flirframe_distribution_right, cmap=cm.gnuplot2)                
        # subplot3.set_title("Distribution right")
        # fig.tight_layout()


        flirPath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\code\各項功能\結果存圖\論文\原始影像_熱影像_分隔去背比較\色彩地圖\\'+'2.jpg'
        plt.axis('off')  
        plt.imsave(flirPath, flirframe_distribution_Left, cmap=cm.gnuplot2)                

        distribution_Left_image = plt.imread(flirPath)
        image = distribution_Left_image.copy()
        image[flimask==0] = [122, 195, 236]

        flirPath1 = r'G:\我的雲端硬碟\Lab\Project\外科溫度\code\各項功能\結果存圖\論文\原始影像_熱影像_分隔去背比較\色彩地圖\\'+'5.jpg'
        plt.axis('off')  
        # plt.imshow(image)
        # plt.show()
        plt.imsave(flirPath1, image) 


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
            ax1.set_ylabel("accumulation")
            if drawLine:
                l1 = ax1.vlines(mean, 0, 7000, linestyles ="-", color="red")
                l2 = ax1.vlines(dataRange + flirframe, 0, 7000, linestyles ="dotted", color="orange")
                l3 = ax1.vlines(conf_intveral, 0, 7000, linestyles ="-.", color="green")
            ax1 = sns.distplot(x, bins = 55, norm_hist=False, kde=False) 
        else:
            ax1.set_ylabel("Distribution")
            if drawLine:
                l1 = ax1.vlines(mean, 0, 0.5, linestyles ="-", color="red")
                l2 = ax1.vlines(dataRange + flirframe, 0, 0.5, linestyles ="dotted", color="orange")
                l2 = ax1.vlines(dataRange - flirframe, 0, 0.5, linestyles ="dotted", color="orange")
                l3 = ax1.vlines(conf_intveral, 0, 0.5, linestyles ="-.", color="green")
            ax1 = sns.distplot(x, bins = 55, norm_hist=True, hist=True, kde=False, fit=stats.norm)                       # 使用SNS繪製，強制擬合常態分佈
            # ax1 = sns.distplot(x, bins = 55, norm_hist=True, hist=True, kde=False)                       # 使用SNS繪製，強制擬合常態分佈
        if drawLine:
            plt.legend(handles=[l1,l2,l3],labels=['mean','mean+'+str(flirframe),str(confidence*100)+'%'],loc='upper right')
            # plt.legend(handles=[l1,l2],labels=['max-4', 'mean-1.5'],loc='upper right')

        fig.tight_layout()
        if pltSavepath:
            print("save at:"+ str(pltSavepath))
            fig.savefig(pltSavepath, dpi=1000, bbox_inches='tight')
            plt.close('all')
        plt.show()


if __name__ == '__main__':
    # imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片\傷口分類\Ischemia FLIR')   # 輸入路徑
    imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\範例圖像\輸入影像')   # 輸入路徑
    saveImgpath = r'結果存圖\論文\原始影像_熱影像_分隔去背比較'
    # saveImgpath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\訓練用數據集\haveBackground\Ischemia FLIR'

    palettes = [cm.jet]                        # 影像調色板

    flirSplit = flir_img_split(imgInputpath, palettes)

    for imgPath in flirSplit.getImglist():
        # print(imgPath)
        imgName = os.path.split(imgPath)[-1]
        savePath = os.path.join(saveImgpath, imgName)

        flirRGB, flirHot = flirSplit.separateNP(imgPath)
        localMin = flirSplit.draw_MeanMin_hist(flirHot, imgName, outputImg=False, savePath=None)            # 畫出背景與患部溫度分佈圖
        
        # flimask, normalObject, hotObject = flirSplit.makeMask(flirHot, localMin, fixmask = False) 
        # flirSplit.drawMask(flirRGB, flirHot, flimask, normalObject, pltSavepath = None)

        flimask, normalObject, hotObject = flirSplit.makeMask(flirHot, localMin, fixmask = True,  outputImg=False, pltSavepath=None) 
        # flirSplit.saveCmapimg(flirHot, flimask, pltSavepath=savePath, imgName=imgName)
        # flirSplit.drawMask(flirRGB, flirHot, flimask, normalObject, pltSavepath = savePath)

        # flirSplit.drawHist(flirHot, flimask, vline=localMin, labelName='Local Minimum', savePath=savePath)


        # flirSplit.saveCmap(normalObject, hotObject, flimask, pltSavepath = None)
        
        flirSplit.deviation_comparison(normalObject, hotObject, flimask)

        break
        


# %%
