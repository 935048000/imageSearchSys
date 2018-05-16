import numpy as np
import argparse
import os
import time


class base (object):
    """
    这是一个基础方法类，封装常用方法。
    """
    
    # 返回目录中指定后缀的文件列表。
    def getFileList(self, path, type):
        return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.' + type)]
    
    # 获取图像文件名字,有文件类别后缀的，无路径。
    def getImageName(self, imagefilepath):
        _temp = imagefilepath.split ("/")[-1]
        return _temp.split ("\\")[-1]
    
    # 获取图像名字,无文件类别后缀。
    def getImageName2(self, imagefile):
        return imagefile.split (".")[0]
    
    # 获取图像信息文件名，“.txt” 为后缀。
    def getImageTxtName(self, imagefile):
        return imagefile + ".txt"

    # 去掉文件路径和后缀
    def getImageName3(self,filename):
        _filename = self.getImageName(filename)
        return self.getImageName2(_filename)

    def plusName(self,imagefile):
        return "H:/datasets/M/" + imagefile + ".JPEG"

    # 获取图像信息
    def getImageInfo(self, image, imagePath):
        _imgInfoList = []
        _imgList = []
        
        def _getImageInfoList(imgList):
            b = base()
            for i in imgList:
                _image = b.getImageName (i)
                _imageName = b.getImageName2 (_image)
                _imageInfoFile = b.getImageTxtName (_imageName)
                _imageInfoFile = imagePath + "/" + _imageInfoFile
                with open (_imageInfoFile, 'r', encoding='utf-8') as f:
                    imageInfo = f.readline ().strip ("\n")
                _imgInfoList.append (imageInfo)
            return _imgInfoList
        
        if type (image) != list:
            _imgList.append (image)
            _imgInfoList = _getImageInfoList (_imgList)
            return _imgInfoList[0]
        else:
            _imgInfoList = _getImageInfoList (image)
            return _imgInfoList

    
    def to16(self,arg):
        return hex(int(arg,2))
    
    
    def to2(self,arg):
        return bin(int(arg,16))

if __name__ == '__main__':
    pass
    a = [[1,"w"],[3,"w3"],[2,"w2"]]
    print(sorted(a,reverse=True))
    print(a.sort(reverse=True))
    print(a)
    # for i in a[:2]:
    #     print(i)
    # for num, index in enumerate (a[:2]):
    #     print(index[1])

