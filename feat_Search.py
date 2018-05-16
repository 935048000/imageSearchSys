import cv2
import scipy as sp
import numpy as np
from base import base
from img_feat import getFeatus,Cosine_distance,getFeatus2
from time import time
from featDB import loadData
from cv2 import imread,imshow,resize,namedWindow,waitKey,cvtColor,COLOR_BGR2RGB,INTER_AREA
from matplotlib import image
from matplotlib import pyplot as plt
from memory_profiler import profile

'''
这是一个特征检索程序，从数据库中检索最相似的图像
This is a feature retrieval program that retrieves the most similar image from a database
'''

# 显示图像
def showImage(imgpath,imgTitle):
    """
    Display image
    :param imgpath:
    :param imgTitle:
    :return:
    """
    # img = imread ("H:/datasets/M/" + imgpath + ".JPEG")
    # namedWindow (imgTitle)
    # imshow (imgTitle, img)
    # waitKey (0)
    
    img = image.imread("H:/datasets/M/" + imgpath + ".JPEG")
    plt.title (imgTitle)
    plt.imshow (img)
    plt.show ()
    
    return 0


def showImage2(imageList, tops):
    """
    Display image
    :param imageList:
    :param tops:
    :return:
    """
    for num, index in enumerate (imageList[:tops]):
        showImage (index[1], "Number %d " % (num + 1))
    return 0


# 显示原图与相似图中间的关键点连线
def showImage3(image1, image2, kp1, kp2, des1, des2):
    """
    Display the key points in the middle of the original image and similar image
    :param image1:
    :param image2:
    :param kp1:
    :param kp2:
    :param des1:
    :param des2:
    :return:
    """
    img1 = cv2.imread (image1, 0)
    img2 = cv2.imread (image2, 0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros ((max (h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]
    
    # 比值判别法
    def _matchScores(matches):
        good = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good.append (m)
        return good
    
    # 计算匹配度
    matches = knn2 (des1, des2)
    # 使用比值判别法，获取指定匹配度的特征描述
    good = _matchScores (matches)
    print (len (good))
    # 可视化特征点之间的连线
    for m in good:
        # 画出要点
        color = tuple ([sp.random.randint (0, 255) for _ in range (3)])
        cv2.line (view, (int (kp1[m.queryIdx].pt[0]), int (kp1[m.queryIdx].pt[1])),
                  (int (kp2[m.trainIdx].pt[0] + w1), int (kp2[m.trainIdx].pt[1])), color)
    cv2.imshow ("view", view)
    cv2.waitKey ()

# 比值判别法
def matchScores(matches):
    """
    Ratio discrimination method
    :param matches:
    :return:
    """
    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append (m)
    return "%.2f" %((len (good)/len (matches))*100)

# KNN匹配
def knn(des1,des2):
    """
    KNN
    :param des1:
    :param des2:
    :return:
    """
    # FLANN 特征
    FLANN_INDEX_KDTREE = 0
    index_params = dict (algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=10)
    flann = cv2.FlannBasedMatcher (index_params, search_params)

    # bf = cv2.BFMatcher (cv2.HAMMING_NORM_TYPE)

    # K-近邻算法 匹配
    matches = flann.knnMatch (des1, des2, k=2)
    return matchScores (matches)

def knn2(des1,des2):
    """
    KNN
    :param des1:
    :param des2:
    :return:
    """
    # FLANN 特征
    FLANN_INDEX_KDTREE = 0
    index_params = dict (algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=10)
    flann = cv2.FlannBasedMatcher (index_params, search_params)

    # bf = cv2.BFMatcher (cv2.HAMMING_NORM_TYPE)

    # K-近邻算法 匹配
    matches = flann.knnMatch (des1, des2, k=2)
    return matches



# @profile(precision=2)
def imgSearch(imgpath):
    """
    image search
    :param imgpath:
    :return:
    """
    t = time ()
    b = base()
    scoreList = []
    # 检索图像名称
    isearchName = b.getImageName3(imgpath)
    isearch = getFeatus(imgpath)
    _feat = loadData("./model/feat_M.npy")
    _file = loadData ("./model/file_M.npy")

    for feat, file in zip (_feat[:1700], _file[:1700]):
        test = []
        _score = knn (isearch, np.array (feat))
        test.append (float (_score))
        test.append (file)
        scoreList.append (test)

    # for i in range(1700):
    #     test = []
    #     _score = knn (isearch,np.array(_feat[i]))
    #     test.append (float(_score))
    #     test.append (_file[i])
    #     scoreList.append(test)
    
    # 排序
    scoreList.sort(reverse=True)
    # 显示
    # showImage (isearchName,"Search Image")
    # showImage2(scoreList,3)
    print("匹配分数：",scoreList[:3])
    print ("检索时间： %.2f s" % (time () - t))
    imgName = b.plusName(scoreList[2][1])
    kp1 = getFeatus2 (imgpath)
    kp2 = getFeatus2 (imgName)
    showImage3(imgpath,imgName,kp1,kp2,isearch,getFeatus(imgName))
    
    return 0

# 测试函数
def searchTest():
    """
    test function
    :return:
    """
    from sys import argv
    imgpath = argv[1]
    print ("需要检索的图像： ", imgpath)
    t = time ()
    b = base ()
    scoreList = []
    isearchName = b.getImageName3 (imgpath)
    isearch = getFeatus (imgpath)
    _feat = loadData ("./model/feat_M.npy")
    _file = loadData ("./model/file_M.npy")

    for feat, file in zip (_feat, _file):
        test = []
        _score = knn (isearch, np.array (feat))
        test.append (float (_score))
        test.append (file)
        scoreList.append (test)
    
    scoreList.sort (reverse=True)
    print ("匹配相似度最高的3个图像和分数：", scoreList[:3])
    print ("检索时间： %.2f s" % (time () - t))
    showImage (isearchName, "Retrieve Images")
    showImage2 (scoreList, 3)
    return 0



def main():
    
    imgSearch("H:/datasets/M/20150630144215059.JPEG")
    
    
    return 0



if __name__ == '__main__':
    pass
    # showImage("19700102125856069","test")
    main()
    # searchTest()
    # img1 = getFeatus("H:/datasets/M/19700102130451860.JPEG")
    # img2 = getFeatus("H:/datasets/M/19700102130451860.JPEG")
    # print(knn(img1,img2))

    # _file = loadData ("./model/file_M.npy")
    # print(_file[256])
    
    # _feat = loadData ("./model/feat_M.npy")
    # isearch = getFeatus ("H:/datasets/M/20150630144423267.JPEG")
    # print (knn (isearch, _feat[256]))