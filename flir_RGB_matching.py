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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
# from tqdm.notebook import tqdm
plt.rcParams['figure.figsize'] = [15, 15]

#%%

# https://tigercosmos.xyz/post/2020/05/cv/image-stitching/s
if __name__ == '__main__':
    imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片')   # 輸入路徑
    # imgInputpath = os.walk(r'G:\我的雲端硬碟\Lab\Project\外科溫度\範例圖像\輸入影像')   # 輸入路徑
    # saveImgpath = r'結果存圖\論文\原始影像_熱影像_去背'
    saveImgpath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\範例圖像\輸出影像'

    palettes = [cm.gnuplot2]                        # 影像調色板

    flirSplit = flir_img_split(imgInputpath, palettes)


    def transform_image(img):
        img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_gray, img, img_rgb

    def SIFT(img):
        siftDetector= cv2.xfeatures2d.SIFT_create() # limit 1000 points
        # siftDetector= cv2.SIFT_create()  # depends on OpenCV version

        kp, des = siftDetector.detectAndCompute(img, None)
        return kp, des

    def plot_sift(gray, rgb, kp):
        tmp = rgb.copy()
        img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img

    def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < threshold*n.distance:
                good.append([m])

        matches = []
        for pair in good:
            matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

        matches = np.array(matches)
        return matches

    def plot_matches(matches, total_img):
        match_img = total_img.copy()
        offset = total_img.shape[1]/2
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
        
        ax.plot(matches[:, 0], matches[:, 1], 'xr')
        ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
        
        ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
                'r', linewidth=0.5)

        plt.show()
    


    def sharpen(img, sigma=100):    
        # sigma = 5、15、25
        blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
        usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

        return usm



    for imgPath in flirSplit.getImglist():
        # print(imgPath)
        imgName = os.path.split(imgPath)[-1]
        savePath = os.path.join(saveImgpath, imgName)

        flirRGB, flirHot = flirSplit.separateNP(imgPath)

        rows, cols, _ = flirRGB.shape
        # rows = int(rows*zoom)
        # cols = int(cols*zoom)

        flirHot = np.stack([flirHot, flirHot, flirHot], axis=2)
        flirHot = cv2.resize(flirHot, (cols, rows), interpolation=cv2.INTER_AREA)
        
        autoNormal = (flirHot - np.amin(flirHot)) / (np.amax(flirHot) - np.amin(flirHot))       # 標準化到 0~1 之間
        flirHot = autoNormal*200
        flirHot = np.uint8(flirHot)
        flirRGB = np.uint8(flirRGB)

        left_gray, left_origin, left_rgb = transform_image(flirHot)
        right_gray, right_origin, right_rgb = transform_image(flirRGB)

        left_gray = sharpen(left_gray)
        # left_gray = cv2.medianBlur(left_gray, 5)
        # left_gray = dst = cv2.GaussianBlur(left_gray, (1, 1), 0)
        left_gray = cv2.adaptiveThreshold(left_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        right_gray = dst = cv2.GaussianBlur(right_gray, (11, 11), 0)
        right_gray = cv2.medianBlur(right_gray, 3)
        right_gray = cv2.adaptiveThreshold(right_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)




        # SIFT only can use gray
        kp_left, des_left = SIFT(left_gray)
        kp_right, des_right = SIFT(right_gray)

        kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
        kp_right_img = plot_sift(right_gray, right_rgb, kp_right)
        total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
        plt.imshow(total_kp)
        
        plt.show

        # matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

        # total_img = np.concatenate((left_rgb, right_rgb), axis=1)
        # plot_matches(matches, total_img) # Good mathces
    
    
        # plt.show
        break
# %%
