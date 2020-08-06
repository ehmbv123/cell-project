# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:02:06 2020

@author: 18333
"""

import numpy as np
import cv2
import os
from tracker import CenterPointTracking
from collections import defaultdict
import functools
from math import sqrt


def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getAllImgPath(path):
    filelist = []
    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            filelist.append(os.path.join(dirpath, fn))
    return filelist

def watershed2(blur):

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opening = cv2.erode(opening,kernel,iterations=5)

    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(opening, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers1 = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers1 + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    bg = np.zeros_like(blur)

    markers3 = cv2.watershed(bg , markers)

    bg[markers3 == -1] = [255, 255, 255]

    return bg

def csNormalize(img):

    dst = np.zeros_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return dst


def task2(img, index, pathDic):
    
    cs = csNormalize(img)

    blur = cv2.GaussianBlur(cs, (3, 3), 0)

    seg = watershed2(blur)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    point = (0, 0)
    square = 0
    num = 0
    divNum = 0
    rects = []
    area = []
    for i in range(len(contours)):

        x, y, w, h = cv2.boundingRect(contours[i])

        s = cv2.contourArea(contours[i])
        area.append(s)

        if s < 10 or s > 1500 or point == (x, y) or square == s:
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
            if pathDic[k]:
                dis = sqrt((v[0] - pathDic[k][-1][0])**2 + (v[1] - pathDic[k][-1][1])**2)
                if dis > 20:
                    continue
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
    show_image(img)
    return pathDic, img

def path_sort(path1, path2):
    path1 = path1.replace('/', '.')
    path2 = path2.replace('/', '.')
    
    p1 = path1.split('.')
    p2 = path2.split('.')
    
    n1 = int(p1[-2])
    n2 = int(p2[-2])
    
    if n1 < n2: return -1
    elif n1 == n2:  return 0
    else:   return 1


if __name__ == '__main__':
    key = 3
    sequence = ['Sequence 1', 'Sequence 2', 'Sequence 3', 'Sequence 4']
    path = "./datasets/Fluo-N2DL-HeLa/" + sequence[key]
    filelist = getAllImgPath(path)
    img_list = []
    output_List = []
    
    index = 0
    ct = CenterPointTracking()
    pathDic = defaultdict(list)
    for f in filelist:
        index += 1
        img = cv2.imread(f)
        img_list.append(img)
        pathDic, output_img = task2(img, index, pathDic)

        output_List.append(output_img)
        cv2.imwrite('./cell2task2/' + sequence[key] + '/' + str(index) + '.jpg', output_img)

    out_root = "./cell2task2/" + sequence[key] + '/'
    output_List = getAllImgPath(out_root)
    output_List = sorted(output_List,key=functools.cmp_to_key(path_sort))
    
    img_o = cv2.imread(output_List[0])
    
    fps = 2
    size = (img_o.shape[1], img_o.shape[0])
    
    videoWriter = cv2.VideoWriter(f'./cell2task2{sequence[key]}_task2.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
    
    for img in output_List[1:]:
        read_img = cv2.imread(img)
        videoWriter.write(read_img)
    
    videoWriter.release()
    cv2.destroyAllWindows()
