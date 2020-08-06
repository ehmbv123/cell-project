import numpy as np
import cv2
import os
from tracker import CenterPointTracking
from collections import defaultdict
import functools

def cv_show(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getAllImgPath1(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return sorted(Filelist)

def polygon_area(polygon):

    n = len(polygon)

    if n < 3:
        return 0

    vectors = np.zeros((n, 2))
    for i in range(0, n):
        vectors[i, :] = polygon[i, :] - polygon[0, :]

    area = 0
    for i in range(1, n):
        area = area + coss_multi(vectors[i - 1, :], vectors[i, :]) / 2

    return area

def coss_multi(v1, v2):

    return v1[0] * v2[1] - v1[1] * v2[0]

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

    bg = np.zeros_like(blur)
    markers3 = cv2.watershed(bg, markers)
    bg[markers3 == -1] = [255, 255, 255]

    return bg

def csMaxMin(img):

    Imax = np.max(img)
    Imin = np.min(img)
    MAX = 255
    MIN = 0
    img_cs = (img - Imin) / (Imax - Imin) * (MAX - MIN) + MIN

    return img_cs.astype("uint8")

def task1(index,img,pathDic):
    cs = csMaxMin(img)
    blur = cv2.GaussianBlur(cs, (7, 7), 0)
    seg = watershed3(blur)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:]

    point = (0, 0)
    square = 0
    num = 0
    rects = []

    for i in range(0, len(contours)):

        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(blur, (x, y), (x + w, y + h), (153, 153, 0), 1)
        s = cv2.contourArea(contours[i])

        if  s < 200 and point == (x, y) and square == s:
            continue
        rects.append((x,y,x+w,y+h))

        cv2.rectangle(img,(x,y), (x+w,y+h), (255, 0, 0), 2, 1)

        point = (x, y)
        square = s
        num += 1

    objects = ct.update(rects)

    if len(pathDic) == 0:
        pathDic[0].append((objects[0][0], objects[0][1]))
        cv2.circle(img, (objects[0][0], objects[0][1]), 10, (0, 255, 0), -1)

    else:
        if 0 in objects.keys():

            pathDic[0].append((objects[0][0], objects[0][1]))
            cv2.circle(img, (objects[0][0], objects[0][1]), 10, (0, 255, 0), -1)

            # start point
            x0, y0 = pathDic[0][0][0], pathDic[0][0][1]
            # last frame point
            x1, y1 = pathDic[0][-2][0], pathDic[0][-2][1]
            # current point
            x2, y2 = pathDic[0][-1][0], pathDic[0][-1][1]

            speed = np.sqrt(np.sum(np.square(x1 - x2) + np.square(y1 - y2)))
            speed = round(speed, 2)

            totalDistance = 0
            pre = 0
            for i, p in enumerate(pathDic[0][1:]):
                if i == 0:
                    totalDistance += np.sqrt(np.sum(np.square(x0 - p[0]) + np.square(x1 - p[1])))
                    pre = (p[0], p[1])
                else:
                    totalDistance += np.sqrt(np.sum(np.square(pre[0] - p[0]) + np.square(pre[1] - p[1])))
            totalDistance = round(totalDistance, 2)

            netDistance = np.sqrt(np.sum(np.square(x0 - x2) + np.square(y0 - y2)))
            netDistance = round(netDistance, 2)

            if netDistance != 0:

                t = totalDistance / netDistance
                confinementRatio = round(t, 2)
                img = cv2.putText(img, "confinement ratio:" + str(confinementRatio), (0, 160),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (100, 200, 200), 2)

            img = cv2.putText(img, "speed:" + str(speed) + "pixels/frame", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (100, 200, 200), 2)
            img = cv2.putText(img, "total distance:" + str(totalDistance), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (100, 200, 200), 2)
            img = cv2.putText(img, "net distance:" + str(netDistance), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (100, 200, 200), 2)


    cv_show(img)
    cv2.imwrite('./task3Img/'+str(index)+".jpg",img)

    return pathDic

def diySort(path1,path2):


    s1 = path1.index("/")
    e1 = path1.index(".")

    s2 = path2.index("/")
    e2 = path2.index(".")

    n1 = int(path1[s1+1:e1])
    n2 = int(path2[s2+1:e2])

    if n1 < n2: return -1
    elif n1 == n2:  return 0
    else:   return 1

if __name__ == '__main__':

    imgPath = 'datasets/PhC-C2DL-PSC/Sequence 1'
    imgList = getAllImgPath1(imgPath)

    pathDic = defaultdict(list)
    frame = cv2.imread("datasets/PhC-C2DL-PSC/Sequence 1/t001.tif")

    selectedBox = cv2.selectROI(frame, False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rects = [(selectedBox[0], selectedBox[1], selectedBox[0] + selectedBox[2], selectedBox[1] + selectedBox[3])]
    ct = CenterPointTracking()
    ct.update(rects)

    for index,i in enumerate(imgList):

         img = cv2.imread(i)
         pathDic = task1(index,img,pathDic)

    imgPath1 = "cell3task1&2"
    imgList1 = getAllImgPath1(imgPath1)
    imgList1 = sorted(imgList1,key=functools.cmp_to_key(diySort))

    img = cv2.imread( imgList1[0])

    fps = 2
    size = (img.shape[1], img.shape[0])

    videoWriter = cv2.VideoWriter('./TestVideo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  fps, size)

    for img in imgList1[1:]:
        read_img = cv2.imread(img)
        videoWriter.write(read_img)

    videoWriter.release()
