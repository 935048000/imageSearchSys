import h5py
import numpy as np
from base import base
from img_feat import getFeatus
from time import time



# 存入文件
def saveData(data,filename):
    if len(data) == 0:
        print("data is None!")
        return 1
    np.save (filename, data)
    return 0

# 读取文件
def loadData(filename):
    _temp = np.load(filename)
    if len(_temp) == 0:
        print("data is None!")
        return 1
    return _temp.tolist()

# 列表组合
def addList(name,feat,info):
    _temp = []
    _temp.append(name)
    _temp.append (feat)
    _temp.append (info)
    return _temp

# 列表组合
def addList2(name,feat,info):
    aList = []
    for i in range(len(name)):
        _temp = []
        _temp.append(name[i])
        _temp.append (feat[i])
        _temp.append (info[i])
        aList.append(_temp)
    
    return aList

# 提取文件列表的所有信息
def getAllList(fileList):
    b = base()
    # 特征数据列表
    _featList = []
    # 图像名称列表
    _fileNameList = []
    # 图像信息列表
    _fileInfoList = []
    
    for i in fileList:
        _featList.append (getFeatus(i))
        _fileNameList.append(b.getImageName2 (b.getImageName (i)))
    
    _fileInfoList = b.getImageInfo(fileList,"D:\data\imageinfo")
    return _fileNameList,_featList, _fileInfoList

# 将数据分批写入文件
def dataWriteFile(num):
    b = base ()
    fileList = b.getFileList ("H:/datasets/M/M%.2d/" %num, "JPEG")
    # infoList = b.getImageInfo (fileList, "H:\datasets\imageinfo")
    t = time()
    _file,_feat,_info = getAllList(fileList)
    print ("M%.2d获取特征时间：" %num, time () - t)
    
    saveData (_file, "./model/file_M%.2d.npy" %num)
    saveData (_feat, "./model/feat_M%.2d.npy" %num)
    saveData (_info, "./model/info_M%.2d.npy" %num)
    
    return 0


def dataWriteFile2():
    b = base ()
    fileList = b.getFileList ("H:/datasets/M/", "JPEG")
    _file, _feat, _info = getAllList (fileList)
    saveData (_file, "./model/file.npy")
    saveData (_feat, "./model/feat.npy")
    saveData (_info, "./model/info.npy")
    return 0

# 数据转换
def data2data():
    test = []
    for num in range(1,22):
        # _file = loadData ("./model/file_M%.2d.npy" %num)
        _feat = loadData ("./model/feat_M%.2d.npy" %num)
        # _info = loadData ("./model/info_M%.2d.npy" %num)
        test += _feat
        

    print(len(test))
    saveData (test, "./model/feat_M.npy")

    return 0

# 测试文件数据是否正常
def testDataFile():
    _file = loadData ("./model/file_M.npy")
    _feat = loadData ("./model/feat_M.npy")
    _info = loadData ("./model/info_M.npy")
    print("读取文件 [正常]")
    # print(len(_file))
    # print(len(_file))
    # print(len(_info))
    if len(_file) == len(_feat) == len(_info):
        print ("文件内容数量 [正常]")
    else:
        print ("文件内容数量 [不正常]")
    # print (_file[304])
    # print (_feat[304])
    # print (_info[304])
    return 0

# 测试数据写入和读取是否正常
def testDataWR():
    # from sys import argv
    # image = argv[1]
    image = "H:/datasets/M/19700102125856069.JPEG"
    feat = getFeatus(image)
    print ("输入图像： ", image)
    try:
        t = time()
        for _ in range(10):
            saveData (feat,"./model/feat_test.npy")
        print ("写入时间： %.2f s" % (time () - t))
        print("写入特征大小：",feat.size)
        print ("特征写入文件 [成功]")
    except:
        print ("特征写入文件 [失败]")

    try:
        t = time ()
        for _ in range (10):
            _feat = loadData ("./model/feat_test.npy")
        print ("读取时间： %.2f s" % (time () - t))
        print ("读取特征大小：", np.array(_feat).size)
        print ("读取特征 [正常]")
    except:
        print ("读取特征 [失败]")
    
    if (feat == np.array(_feat)).all:
        print("特征存取 [正常]")
    else:
        print ("特征存取 [失败]")

    return 0

if __name__ == '__main__':
    pass
    # b = base()

    # data2data()

    # _feat = loadData ("./model/feat_M.npy")


    # import feat_Search
    # print(feat_Search.knn(np.array(_feat[0]),np.array(_feat[0])))
    
    # print(Cosine_distance(np.array(_feat[0]),np.array(_feat2[4])))
    # img1 = img_feat.getHash2(np.float32 (_feat[0]))
    # img2 = img_feat.getHash (np.float32 (_feat[5]))
    # print(img1)
    # print(hex(int(img1,2)))
    # print (img_feat.Cosine_distance2 (img1,img2))
    # print (_info[0],_info[5])
    # print(test2)
    
    # for i in range(10):
    #     saveData(hex(int(img1,2)),"./model/16test.txt")

    testDataWR()