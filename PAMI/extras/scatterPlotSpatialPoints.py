import matplotlib.pyplot as _plt
import pandas as _pd
import validators as _validators
from urllib.request import urlopen as _urlopen


class scatterPlotSpatialPoints:

    def __init__(self, iFile, sep = '\t'):

        self._iFile = iFile
        self._sep = sep

    def _scanningPoints(self):

        points = {}
        if isinstance(self._iFile, _pd.DataFrame):
            x, y = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'x' in i:
                x = self._iFile['x'].tolist()
            if 'y' in i:
                y = self._iFile['y'].tolist()
            for i in range(len(y)):
                points[x[i]] = y[i]

        if isinstance(self._iFile, str):
            if _validators.url(self._iFile):
                data = _urlopen(self._iFile)
                for line in data:
                    line = line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    points[temp[0]] = points[temp[1]]
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            points[temp[0]] = temp[1]
                except IOError:
                    print("File Not Found")
                    quit()
        return points

    def scatterPlotSpatialPoints(self):
        points = self._scanningPoints()
        keys = [i for i in points.keys()]
        values = [i for i in points.values()]
        _plt.scatter(keys, values, c="Red")
        _plt.xlabel("X-axis")
        _plt.ylabel("Y-axis")
        _plt.show()
        print("Scatter Plot is generated")


if __name__ == '__main__':
    ab = scatterPlotSpatialPoints(iFile = '/Users/Likhitha/Downloads/spatial_T10I4D100K.csv', sep = ',')
    ab.scatterPlotSpatialPoints()