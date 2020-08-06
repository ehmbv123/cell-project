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

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(opening, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

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

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 检索模式为树形cv2.RETR_TREE
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
        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 0, 128), 1)

        for objectID, centroid in pathDic.items():
            if len(centroid) > 1:
                pre = centroid[0]
                for i in centroid[1:]:
                    cv2.line(img, (pre[0], pre[1]), (i[0],i[1]), (0, 255, 0), 2, 2)
                    pre = i
            else:
                cv2.circle(img, (centroid[0][0], centroid[0][1]), 4, (0, 255, 0), -1)

    img = cv2.putText(img, "cell number:" + str(num), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 69, 0), 2)

    cv2.imwrite('./cell3task1&2/'+str(index)+".jpg",img)
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

    imgPath = 'datasets/PhC-C2DL-PSC/Sequence 1'
    imgList = getAllImgPath1(imgPath)

    ct = CenterPointTracking()
    pathDic = defaultdict(list)
    
    for index,i in enumerate(imgList):

         img = cv2.imread(i)

         pathDic = task1(index,img,pathDic)

    imgPath1 = "cell3task1&2"
    imgList1 = getAllImgPath1(imgPath1)
    imgList1 = sorted(imgList1,key=functools.cmp_to_key(diySort))

    img = cv2.imread( imgList1[0])


    fps = 2
    size = (img.shape[1], img.shape[0])

    videoWriter = cv2.VideoWriter('./cell3task1&2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  fps, size)

    for img in imgList1[1:]:
        read_img = cv2.imread(img)
        videoWriter.write(read_img)

    videoWriter.release()


