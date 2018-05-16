import cv2
import scipy as sp
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist,cdist
from time import time
# from .b import base
import base
from scipy.misc import imread
from sys import argv

'''
This is a feature extraction program that contains feature extraction procedures.
这是一个特征提取程序，里面包含特征提取的相关程序。
'''

# 归一化
def normalization(vec):
    return vec / LA.norm (vec)

# 输入灰度图，返回hash
def getHash(image):
    """
    Enter grayscale and return hash.
    :param image:
    :return:
    """
    avreage = np.mean (image)
    hash = []
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            if image[i, j] > avreage:
                hash.append (1)
            else:
                hash.append (0)
    return hash

# 输入灰度图，返回hash
def getHash2(image):
    """
    Enter grayscale and return hash.
    :param image:
    :return:
    """
    avreage = np.mean (image)
    hash = ""
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            if image[i, j] > avreage:
                # hash.append (1)
                hash += "1"
            else:
                # hash.append (0)
                hash += "0"
    return hash

# 计算余弦距离
def Cosine_distance(arg1,arg2):
    """
    Cosine distance
    :param arg1:
    :param arg2:
    :return:
    """
    AVGList = []
    for i in range(arg1.shape[0]):
        dist = pdist (np.vstack ([arg1[i], arg2[i]]), 'cosine')
        AVGList.append(dist)
    AVG = np.array(AVGList)
    # print(AVG.shape)
    return "%.2f" %((1 - AVG.mean ()) * 100)
    # dist = pdist (np.vstack ([arg1, arg2]), 'cosine')
    # return dist

# 计算余弦距离
def Cosine_distance2(arg1,arg2):
    """
    Cosine distance
    :param arg1:
    :param arg2:
    :return:
    """
    dist = pdist (np.vstack ([arg1, arg2]), 'cosine')
    return dist

def cos_cdist(vector,matrixDB):
    """
    Cosine distance
    :param vector:
    :param matrixDB:
    :return:
    """
    # getting cosine distance between search image and images database
    # 计算待搜索图像与数据库图像的余弦距离
    v = vector.reshape (1, -1)
    return cdist (matrixDB, v, 'cosine').reshape (-1)


# 可视化,显示图像
def showImage(img1,img2,kp1,kp2,good):
    """
    Visualize, display images.
    :param img1:
    :param img2:
    :param kp1:
    :param kp2:
    :param good:
    :return:
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros ((max (h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]

    for m in good:
        # 画出要点
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple ([sp.random.randint (0, 255) for _ in range (3)])
        # print 'kp1,kp2',kp1,kp2
        cv2.line (view, (int (kp1[m.queryIdx].pt[0]), int (kp1[m.queryIdx].pt[1])),
                  (int (kp2[m.trainIdx].pt[0] + w1), int (kp2[m.trainIdx].pt[1])), color)

    cv2.imshow ("view", view)
    cv2.waitKey ()

# 获取特征值
def getFeatus(img):
    """
    Gets the eigenvalue description.
    :param img:
    :return:
    """
    ints = 128
    _img = cv2.imread (img, 0)
    sift = cv2.xfeatures2d.SIFT_create (ints)
    kp, des = sift.detectAndCompute (_img, None)
    return des[:ints]

# 获取关键点
def getFeatus2(img):
    """
    Get key points
    :param img:
    :return:
    """
    ints = 128
    _img = cv2.imread (img, 0)
    sift = cv2.xfeatures2d.SIFT_create (ints)
    kp, des = sift.detectAndCompute (_img, None)
    return kp

# # 获取特征值
# def getFeatus2(image_path,vector_size):
#     image = imread (image_path, mode='RGB')
#     alg = cv2.xfeatures2d.SIFT_create ()
#     kps = alg.detect (image)
#     kps = sorted (kps, key=lambda x: -x.response)[:vector_size]
#     kps, dsc = alg.compute (image, kps)
#     dsc = dsc.flatten ()
#     needed_size = (vector_size * 64)
#     if dsc.size < 32:
#         dsc = np.concatenate ([dsc, np.zeros (needed_size - dsc.size)])
#     return dsc

# 特征提取测试
def featTest():
    """
    Feature extraction test
    :return:
    """
    t = time ()
    image = argv[1]
    img = getFeatus(image)
    print("输入图像： ",image)
    print("特征值形状：",img.shape)
    print ("特征提取耗时： %.2f s" % (time () - t))
    return 0

if __name__ == '__main__':
    testingset = "H:/datasets/testingset/"
    trainingset = "H:/datasets/trainset/"
    b = base.base ()

    # img1 = getFeatus ("H:/datasets/M/19700102125856069.JPEG")
    # img2 = getFeatus ("H:/datasets/M/19700102125908465.JPEG")
    # img3 = getFeatus ("H:/datasets/M/19700102130018220.JPEG")
    # img3 = getFeatus ("H:/datasets/M/19700102125912230.JPEG")

    # t = time ()
    # img11 = getFeatus2 ("H:/datasets/M/19700102125856069.JPEG",128)
    # img12 = getFeatus2 ("H:/datasets/M/19700102125908465.JPEG",128)
    # img13 = getFeatus2 ("H:/datasets/M/19700102130018220.JPEG",128)
    # print ("时间：", time () - t)
    # print(img11.shape)
    # test12 = Cosine_distance2 (img11, img12)
    # test13 = Cosine_distance2 (img11, img13)
    # test14 = Cosine_distance2 (img12, img13)
    # print (test12)
    # print (test13)
    # print (test14)

    # print (img1.shape)
    # img1_1 = img1.reshape (1, -1)
    # print (img1_1.shape)
    # print (img1_1)
    

    # t = time ()
    
    # print (img2.shape)
    # print (img3.shape)
    # print(img1)
    # t1 = getHash (np.float32 (img1))
    # t2 = getHash (np.float32 (img2))
    # t3 = getHash (np.float32 (img3))
    
    # _test = getHash2 (np.float32 (img1))
    # test16 = b.to16 (_test)
    # print(test16)
    # print(len(t1))
    
    
    # test2 = Cosine_distance2(t1,t2)
    # print ("时间：", time () - t)
    # test3 = Cosine_distance2 (t1, t3)
    # test4 = Cosine_distance2 (t2, t3)
    # print(test2)
    # print(test3)
    # print(test4)
    # print()
    # print (Cosine_distance (img1, img1))
    # print (Cosine_distance (img1, img2))
    # print (Cosine_distance (img1, img3))
    # print (Cosine_distance (img2, img3))
    #
    
    # t = time ()
    # print(knn(img2,img1))
    # print ("时间：", time () - t)

    # print(img1.shape)
    # print (knn (img1, img1))
    # print (knn (img1, img2))
    # print (knn (img1, img3))
    # print (knn (img2, img3))
    
    # 特征值提取测试
    featTest()
