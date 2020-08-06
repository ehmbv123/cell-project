# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:33:45 2020

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

    # markers3 = cv2.watershed(blur, markers)
    markers3 = cv2.watershed(bg , markers)

    bg[markers3 == -1] = [255, 255, 255]

    return bg

def csNormalize(img):

    dst = np.zeros_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return dst

def task3(img, index, pathDic, ct):
    index += 1
    cs = csNormalize(img)

    blur = cv2.GaussianBlur(cs, (3, 3), 0)

    seg = watershed2(blur)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    point = (0, 0)
    square = 0
    rects = []
    
    for i in range(len(contours)):

        x, y, w, h = cv2.boundingRect(contours[i])
        s = cv2.contourArea(contours[i])

        if s < 10 or s > 1500 or point == (x, y) or square == s:
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
            print(totalDistance)

            netDistance = np.sqrt(np.sum(np.square(x0 - x2) + np.square(y0 - y2)))
            netDistance = round(netDistance, 2)

            confinementRatio = round(totalDistance/netDistance,2)

            img = cv2.putText(img, "speed:" + str(speed) + "pixels/frame" , (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)
            img = cv2.putText(img, "total distance:" + str(totalDistance) , (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (100, 200, 200), 2)
            img = cv2.putText(img, "net distance:" + str(netDistance), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)
            img = cv2.putText(img, "confinement ratio:" + str(confinementRatio), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (100, 200, 200), 2)
    
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
    key = 1
    sequence = ['Sequence 1', 'Sequence 2', 'Sequence 3', 'Sequence 4']
    path = "datasets/Fluo-N2DL-HeLa/" + sequence[key]
    filelist = getAllImgPath(path)
    img_list = []
    output_List = []
    
    frame = cv2.imread(path + '/t000.tif')
    selectedBox = cv2.selectROI(frame, False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rects = [(selectedBox[0], selectedBox[1], selectedBox[0] + selectedBox[2], selectedBox[1] + selectedBox[3])]

    ct = CenterPointTracking()
    ct.update(rects)
    
    index = 0
    pathDic = defaultdict(list)
    for f in filelist:
        index += 1
        img = cv2.imread(f)
        img_list.append(img)
        pathDic, output_img = task3(img, index, pathDic, ct)
        cv2.imwrite('./cell2task3/' + sequence[key] + '/' + str(index) + '.jpg', output_img)


    out_root = "./cell2task3/" + sequence[key] + '/'
    output_List = getAllImgPath(out_root)
    img_o = cv2.imread(output_List[0])
    
    fps = 1
    size = (img_o.shape[1], img_o.shape[0])
    
    videoWriter = cv2.VideoWriter(f'./{sequence[key]}_task3.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    for img in output_List[1:]:
        read_img = cv2.imread(img)
        videoWriter.write(read_img)
    
    videoWriter.release()
    cv2.destroyAllWindows()