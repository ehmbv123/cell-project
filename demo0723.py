import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

from skimage.filters import meijering
from collections import defaultdict
from skimage import img_as_ubyte

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import imutils


def getAllImgPath(categoryNum,sequenceNum):

    cellCategory = ['', '/DIC-C2DH-HeLa', '/Fluo-N2DL-HeLa', '/PhC-C2DL-PSC']
    sequence = ['', '/Sequence 1', '/Sequence 2', '/Sequence 3', '/Sequence 4']

    path = './COMP9517 20T2 Group Project Image Sequences' + cellCategory[categoryNum] + sequence[sequenceNum]

    # print(path)
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append(filename)
    # print(Filelist)
    return sorted(Filelist)

def cv_show(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 拉伸1: 通过最大最小值拉伸
# https://blog.csdn.net/saltriver/article/details/79677199
def csMaxMin(img):

    Imax = np.max(img)
    Imin = np.min(img)
    MAX = 255
    MIN = 0
    img_cs = (img - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
    # cv_show(np.hstack([img, img_cs.astype("uint8")]))
    # cv_show( img_cs.astype("uint8") )
    return img_cs.astype("uint8")
# 返回直方图
def drawHistGraph(img):
    hist = cv2.calcHist([img],
                        [0],  # 使用的通道
                        None,  # 没有使用mask
                        [256],  # HistSize
                        [0.0, 255.0])  # 直方图柱的范围
    plt.figure()  # 新建一个图像
    plt.title("Histogram")  # 图像的标题
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签
    plt.plot(hist)  # 画图
    plt.xlim([0, 256])  # 设置x坐标轴范围
    plt.show()  # 显示图像

# 拉伸2: 直方图归一化
# https://blog.csdn.net/qq_40755643/article/details/84032773
def csNormalize(img):

    dst = np.zeros_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # cv_show(np.hstack([img, dst]))

    return dst

# 拉伸3: 直方图均衡化
def csEqualizeHist(img):

    # 原来图像的直方图 还没均衡化的
    # drawHistGraph(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
    eq = cv2.equalizeHist(img)
    eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)  # 转化成灰度图
    # 均衡化后的直方图
    # drawHistGraph(dst)
    return eq

# 简单阈值 选取一个全局阈值，然后就把整幅图像分成了非黑即白的二值图像 手动调整
# https://blog.csdn.net/jjddss/article/details/72841141
def thresholdSimple(img):
    print(img)
    ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    # cv_show(thresh)
    return thresh

# 自适应阈值
# 自适应阈值可以看成一种局部性的阈值，通过规定一个区域大小，
# 比较这个点与区域大小里面像素点的平均值（或者其他特征）的大小关系确定这个像素点是属于黑或者白
def thresholdAdap(img):

    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 换行符号
    return thresh

# 大津法
def thresholdOTSU(img):

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# 针对第二类图像的分水岭算法
# https://www.jianshu.com/p/d0a812db9eae
def watershed2(blur):

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 转化成灰度图

    # gray = cv2.dilate(gray,kernel,iterations=)
    # 二值化处理
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = cv2.erode(opening,kernel,iterations=1)
    # cv_show(opening)
    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # 寻找未知的区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers1 = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers1 + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    bg = np.zeros_like(blur)

    # markers3 = cv2.watershed(blur, markers)
    markers3 = cv2.watershed(bg , markers)
    # cv_show(img)
    # blur[markers3 == -1] = [0, 0, 255]
    bg[markers3 == -1] = [255, 255, 255]
    # cv_show(bg)
    return bg

# 针对第三类图像的分水岭算法
def watershed3(blur):

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 转化成灰度图

    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 换行符号
    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # cv_show(thresh)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv_show(np.hstack([thresh, opening]))

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    # cv_show(sure_bg)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # 寻找未知的区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers1 = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers1 + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # markers3 = cv2.watershed(blur, markers)
    # cv_show(img)
    # blur[markers3 == -1] = [0, 0, 255]
    # cv_show(blur)

    bg = np.zeros_like(blur)

    # markers3 = cv2.watershed(blur, markers)
    markers3 = cv2.watershed(bg, markers)
    # cv_show(img)
    # blur[markers3 == -1] = [0, 0, 255]
    bg[markers3 == -1] = [255, 255, 255]
    # cv_show(bg)
    return bg

    # return blur

# 针对第一类图像的分水岭算法
# 效果不好
def watershed1(blur):

    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
    # cv_show(blur)
    print(gray.shape)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # 换行符号
    # ret, thresh = cv2.threshold(blur, 142, 255, cv2.THRESH_BINARY)

    thresh = 255 - thresh
    cv_show( np.hstack([gray,thresh]) )

    kernel = np.ones((3, 3), np.uint8)

    # i = cv2.erode(thresh,kernel,iterations=1)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = cv2.erode(opening, kernel, iterations=1)
    # i = cv2.dilate(i,kernel,iterations=1)

    # cv_show(np.hstack([thresh, i]))

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    cv_show(sure_bg)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # 寻找未知的区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers1 = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers1 + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers3 = cv2.watershed(blur, markers)
    # cv_show(img)
    blur[markers3 == -1] = [0, 0, 255]
    cv_show(blur)
    return blur

# meijering
def task11Cell1(img,path):

    img = meijering(img, 10, black_ridges=True, mode='nearest', cval=0)
    # print( img.dtype )
    img = img_as_ubyte(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv_show(img)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv_show(blur)

    kernel = np.ones((3, 3), np.uint8)
    #
    blur = cv2.erode(blur,kernel,iterations=3)
    thresh = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv_show(opening)
    #
    # print(opening*255)
    # # cv_show(opening*255)
    # 二值化处理
    # ret, thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)
    #
    # cv_show(thresh)

    opening  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    erobe = cv2.erode(opening, kernel, iterations=1)
    # cv_show(erobe)

    # canny = cv2.Canny(img, 50, 200)
    # cv_show(canny)

    contours, hierarchy = cv2.findContours(erobe,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:]

    # print(  [ cv2.contourArea(i) for i in contours ])
    # print( cv2.contourArea(contours[-1]) )

    cell = defaultdict(list)
    point = (0, 0)
    square = 0
    index = 0

    for i in range(0, len(contours)):
        if abs( cv2.contourArea(contours[i]) - 10000 ) <= 8000:
            # 最小外接矩形
            rect = cv2.minAreaRect(contours[i])

            # 矩形四个角点取整
            box = np.int0(cv2.boxPoints(rect))

            x = np.round(rect[0][0]).astype("int")
            y = np.round(rect[0][1]).astype("int")

            s = polygon_area(box)
            if point == (x, y) and square == s:
                continue

            point = (x, y)
            square = polygon_area(box)
            cell[index].append(((x, y), s))
            index += 1
            cv2.drawContours(blur, [box], 0, (255, 0, 0), 2)

    if len(path) == 0:
        path = cell

    else:

        for k1, v1 in cell.items():

            for k2, v2 in path.items():
                if (abs(v1[-1][1] - v2[-1][1]) <= 200) and (abs(v1[-1][0][0] - v2[-1][0][0]) <= 5) and (
                        abs(v1[-1][0][1] - v2[-1][0][1]) <= 5):
                    path[k2].extend(v1)

        # break
    # cv_show(blur)
    return blur,path

def task11Cell2(img,path):

    # 进行对比度拉伸
    cs = csNormalize(img)

    # # 高斯模糊
    blur = cv2.GaussianBlur(cs, (5, 5), 0)
    # cv_show(np.hstack([cs, blur]))

    # 分水岭找边缘
    seg = watershed2(blur)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:]

    cell  = defaultdict(list)
    point = (0,0)
    square = 0
    index = 0

    for i in range(0, len(contours)):

        # 外接矩形
        # x, y, w, h = cv2.boundingRect(contours[i])
        # cv2.rectangle(blur, (x, y), (x + w, y + h), (153, 153, 0), 1)

        # 最小外接矩形
        rect = cv2.minAreaRect(contours[i])
        # 矩形四个角点取整
        box = np.int0(cv2.boxPoints(rect))

        x = np.round(rect[0][0]).astype("int")
        y = np.round(rect[0][1]).astype("int")

        s = polygon_area(box)
        if point ==  (x,y) and square == s :
            continue

        point = (x,y)
        square = polygon_area(box)
        cell[index].append( ((x,y),s) )
        index += 1
        cv2.drawContours(blur, [box], 0, (255, 0, 0), 2)


    if len(path) == 0:
        path = cell

    else:

        for k1,v1 in cell.items():

            for k2,v2 in path.items():
                if ( abs(v1[-1][1] - v2[-1][1]) <= 200 ) and ( abs(v1[-1][0][0] - v2[-1][0][0]) <= 5 ) and ( abs(v1[-1][0][1] - v2[-1][0][1]) <= 5 ):
                    path[k2].extend( v1 )



    # print(cell.keys())


    return blur,path,len(cell.keys())

def task11Cell3(img,path):

    cs = csMaxMin(img)

    # 高斯模糊
    blur = cv2.GaussianBlur(cs, (7, 7), 0)
    # cv_show(np.hstack([cs, blur]))

    seg = watershed3(blur)
    # cv_show(seg)

    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    # print(gray.dtype)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:]
    # print(contours)

    cell = defaultdict(list)
    point = (0, 0)
    square = 0
    index = 0

    for i in range(0, len(contours)):

        # 外接矩形
        # x, y, w, h = cv2.boundingRect(contours[i])
        # cv2.rectangle(blur, (x, y), (x + w, y + h), (153, 153, 0), 1)

        # 最小外接矩形
        rect = cv2.minAreaRect(contours[i])
        # 矩形四个角点取整
        box = np.int0(cv2.boxPoints(rect))

        x = np.round(rect[0][0]).astype("int")
        y = np.round(rect[0][1]).astype("int")

        s = polygon_area(box)
        if point == (x, y) and square == s:
            continue

        point = (x, y)
        square = polygon_area(box)
        cell[index].append(((x, y), s))
        index += 1
        cv2.drawContours(blur, [box], 0, (255, 0, 0), 2)

    if len(path) == 0:
        path = cell

    else:
        # print(cell)
        for k1, v1 in cell.items():
            for k2, v2 in path.items():
                if (abs(v1[-1][1] - v2[-1][1]) <= 200) and (abs(v1[-1][0][0] - v2[-1][0][0]) <= 5) and (
                        abs(v1[-1][0][1] - v2[-1][0][1]) <= 5):
                    path[k2].extend(v1)

        # print(path)
        # print(cell)

    # print(  )
    # print(polygon_area(np.array(allPoint)))
    return blur, path ,len(cell.keys())

def drawPath(img,path):

    img = csNormalize(img)
    # print(path)
    for k, v in path.items():
        pre = 0
        for ps in v:
            # print(ps)
            if pre == 0:
                pre = ps[0]
                cv2.circle(img, pre, 2, (0, 0, 255), 4)
            else:
                cv2.line(img, pre, ps[0], (0, 0, 255), 4)
                pre = ps[0]
    cv_show(img)


if __name__ == '__main__':

    categoryNum = 2
    sequenceNum = 1

    imgPathList = getAllImgPath(categoryNum,sequenceNum)

    task1Path = defaultdict(list)

    for imgPath in imgPathList[:]:

        img = cv2.imread(imgPath)
        # cv_show(img)

        if categoryNum == 1:

            img,path = task11Cell1(img,task1Path)
            task1Path = path
            # cv_show(img)

        elif categoryNum == 2:

            outline,path,num = task11Cell2(img,task1Path)
            task1Path = path

            img = cv2.putText(outline, "cell number:"+str(num), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 200, 200), 2)

            cv_show(outline)
            # print(num)
            # print()
            # print( path )

        elif categoryNum == 3:
            outline,path,num = task11Cell3(img,task1Path)
            task1Path = path

            img = cv2.putText(outline, "cell number:" + str(num), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                              (100, 200, 200), 2)

            cv_show(outline)

    img = cv2.imread(imgPathList[0])
    drawPath(img, task1Path)

    # print( len(task1Path.keys()) )


# 进行mean—shift滤波
        # shifted = cv2.pyrMeanShiftFiltering(cs, 5, 51)
        # cv_show(np.hstack([blur, shifted]))
