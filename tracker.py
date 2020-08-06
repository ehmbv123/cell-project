# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CenterPointTracking():

    def __init__(self, maxDisapperFrame=2):

        self.nextId = 0
        self.items = OrderedDict()
        self.disappear = OrderedDict()
        self.maxDisapperFrame = maxDisapperFrame

    def addNewPoint(self, centrePoint):

        self.items[self.nextId] = centrePoint
        self.nextId += 1
        self.disappear[self.nextId] = 0

    def deletePoint(self, id):

        del self.items[id]
        del self.disappear[id]

    def update(self, rectList):

        if len(rectList) == 0:

            keyList = list( self.disappear.keys() )

            for id in keyList:
                self.disappear[id] += 1
                if self.disappear[id] > self.disappear:
                    self.deletePoint(id)

            return self.items

        newCentralPoint = np.zeros((len(rectList), 2), dtype="int")

        for (i, (x, y, w, h)) in enumerate(rectList):

            newX = (x + w) / 2.0
            newY = (y + h) / 2.0

            newCentralPoint[i] = (int(newX), int(newY))

        if len(self.items) != 0:

            keyId = self.items.keys()
            idList = list(keyId)

            value = self.items.values()
            objectCentroids = list(value)

            distance = dist.cdist(np.array(objectCentroids), newCentralPoint)

            allRows = distance.min(axis=1).argsort()

            allColumns = distance.argmin(axis=1)[allRows]

            rowSet = set()
            colSet = set()

            for (r, c) in zip(allRows, allColumns):

                if r in rowSet:
                    continue

                elif c in colSet:
                    continue

                id = idList[r]
                self.items[id] = newCentralPoint[c]
                self.disappear[id] = 0

                rowSet.add(r)
                colSet.add(c)

            newRow = set(range(0, distance.shape[0])).difference(rowSet)
            newColumn = set(range(0, distance.shape[1])).difference(colSet)

            if distance.shape[0] >= distance.shape[1]:

                for row in newRow:

                    objectID = idList[row]
                    self.disappear[objectID] += 1

                    if self.disappear[objectID] > self.maxDisapperFrame:
                        self.deletePoint(objectID)

            else:
                for col in newColumn:
                    self.addNewPoint(newCentralPoint[col])
        else:

            end = len(newCentralPoint)

            for i in range(0, end):
                self.addNewPoint(newCentralPoint[i])

        return self.items