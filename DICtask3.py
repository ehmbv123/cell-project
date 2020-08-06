import numpy as np
import cv2
import os
from tracker import CenterPointTracking
from collections import defaultdict

def cv_show(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def coss_multi(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def getAllImgPath(path,condition=''):

    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if filename.startswith(condition):
                Filelist.append(os.path.join(home, filename))

    return sorted(Filelist)

def task3(index,mask,img,pathDic,ct):

    index += 1

    h = mask.shape[0]
    w = int(mask.shape[1] / 2)

    mask = mask[:h, :w]

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    point = (0, 0)
    square = 0
    rects = []

    for i in range(0, len(contours)):

        x, y, w, h = cv2.boundingRect(contours[i])
        s = cv2.contourArea(contours[i])

        if  s < 200 and point == (x, y) and square == s:
            continue
        rects.append((x,y,x+w,y+h))

        point = (x, y)
        square = s

    objects = ct.update(rects)

    if len(pathDic) == 0:
        pathDic[0].append( (objects[0][0],objects[0][1]) )
        cv2.circle(img, (objects[0][0], objects[0][1]), 10, (0, 255, 0), -1)

    else:
        if 0 in objects.keys():

            pathDic[0].append( (objects[0][0],objects[0][1]) )
            cv2.circle(img, (objects[0][0], objects[0][1]),10, (0, 255, 0), -1)

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
            for i,p in enumerate(pathDic[0][1:]):
                if i == 0:
                    totalDistance += np.sqrt(np.sum(np.square(x0 - p[0]) + np.square(x1 - p[1])))
                    pre = (p[0],p[1])
                else:
                    totalDistance += np.sqrt(np.sum(np.square(pre[0] - p[0]) + np.square(pre[1] - p[1])))
            totalDistance = round(totalDistance,2)

            netDistance = np.sqrt(np.sum(np.square(x0 - x2) + np.square(y0 - y2)))
            netDistance = round(netDistance, 2)

            confinementRatio = round(totalDistance/netDistance,2)

            img = cv2.putText(img, "speed:" + str(speed) + "pixels/frame" , (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)
            img = cv2.putText(img, "total distance:" + str(totalDistance) , (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (100, 200, 200), 2)
            img = cv2.putText(img, "net distance:" + str(netDistance), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)
            img = cv2.putText(img, "confinement ratio:" + str(confinementRatio) , (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (100, 200, 200), 2)

    cv2.imwrite('./cell1task3/'+str(index)+".jpg",img)
    cv_show(img)
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
    # return int(path1[s1+1:e1]) > int(path2[s2+1:e2])

if __name__ == '__main__':

    maskPath = 'datasets/DIC-C2DH-HeLa/01_VIZ'
    condition = 'network'
    maskList = getAllImgPath(maskPath, condition)

    imgPath = 'datasets/DIC-C2DH-HeLa/01'
    imgList = getAllImgPath(imgPath)

    pathDic = defaultdict(list)

    frame = cv2.imread("datasets/DIC-C2DH-HeLa/01/t001.tif")
    selectedBox = cv2.selectROI(frame, False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rects = [(selectedBox[0], selectedBox[1], selectedBox[0] + selectedBox[2], selectedBox[1] + selectedBox[3])]

    ct = CenterPointTracking()
    ct.update(rects)

    for index, i in enumerate(zip(maskList, imgList)):

        mask = cv2.imread(i[0])

        img = cv2.imread(i[1])

        path = task3(index, mask, img, pathDic, ct)


    imgPath = './cell1task3'
    imgList = getAllImgPath(imgPath)

    img = cv2.imread(imgList[0])

    fps = 1
    size = (img.shape[1], img.shape[0])

    videoWriter = cv2.VideoWriter('./cell1task3.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                                  fps, size)

    for img in imgList[1:]:
        read_img = cv2.imread(img)
        videoWriter.write(read_img)

    videoWriter.release()







