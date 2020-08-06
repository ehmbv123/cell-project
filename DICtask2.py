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

def coss_multi(v1, v2):

    return v1[0] * v2[1] - v1[1] * v2[0]

def getAllImgPath(path,condition=''):

    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if filename.startswith(condition):
                Filelist.append(os.path.join(home, filename))

    return sorted(Filelist)

def task12(index,mask,img,pathDic,ct):

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
    num = 0
    rects = []
    divNum = 0

    for i in range(0, len(contours)):

        x, y, w, h = cv2.boundingRect(contours[i])

        s = cv2.contourArea(contours[i])

        if  s < 200 and point == (x, y) and square == s:
            continue
        rects.append((x,y,x+w,y+h))

        if int(np.average(img[y:y+h,x:x+w])) > 122.0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            divNum += 1
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)

        point = (x, y)
        square = s
        num += 1

    objects = ct.update(rects)

    if len(pathDic) == 0:
        for i,v in objects.items():
            pathDic[i].append( (objects[i][0],objects[i][1]) )
    else:

        objectsKey = objects.keys()

        for k,v in objects.items():
            pathDic[k].append(v)

        pathDic1 = pathDic.copy()

        for k,v in pathDic.items():
            if k not in objectsKey:
                del pathDic1[k]

        pathDic = pathDic1

    for (objectID, centroid) in objects.items():

        text = "ID {}".format(objectID)
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for objectID, centroid in pathDic.items():
            if len(centroid) > 1:
                pre = centroid[0]
                for i in centroid[1:]:
                    cv2.line(img, (pre[0], pre[1]), (i[0],i[1]), (0, 255, 0), 2, 2)
                    pre = i
            else:
                cv2.circle(img, (centroid[0][0], centroid[0][1]), 4, (0, 255, 0), -1)

    img = cv2.putText(img, "cell number:" + str(num), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)
    img = cv2.putText(img, "division cell number:" + str(divNum), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)

    cv2.imwrite('./cell1task2/'+str(index)+".jpg",img)
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

if __name__ == '__main__':

    maskPath = 'datasets/DIC-C2DH-HeLa/01_VIZ'
    condition = 'network'
    maskList = getAllImgPath(maskPath,condition)

    imgPath = 'datasets/DIC-C2DH-HeLa/01'
    imgList = getAllImgPath(imgPath)

    ct = CenterPointTracking()
    pathDic = defaultdict(list)

    for index, i in enumerate(zip(maskList[:], imgList[:])):

        mask = cv2.imread(i[0])

        img = cv2.imread(i[1])

        pathDic = task12(index,mask,img,pathDic,ct)
        # print(task1Path)

    imgPath = 'cell1task2'
    imgList = getAllImgPath(imgPath)

    imgList = sorted(imgList,key=functools.cmp_to_key(diySort))

    img = cv2.imread(imgList[0])

    fps = 2
    size = (img.shape[1], img.shape[0])

    videoWriter = cv2.VideoWriter('./cell1task2.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                  fps, size)

    for img in imgList[1:]:
        read_img = cv2.imread(img)
        videoWriter.write(read_img)

    videoWriter.release()
