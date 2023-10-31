# scatterPlotSpatialPoints is used to convert the given data and plot the points.
#
#   **Importing this algorithm into a python program**
#   --------------------------------------------------------
#
#   from PAMI.extras.syntheticDataGenerator import scatterPlotSpatialPoints as plt
#
#   obj = plt.scatterPlotSpatialPoints(iFile, "\t")
#
#   obj.save()
#
#

__copyright__ = """
 Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import matplotlib.pyplot as _plt
import pandas as _pd
from urllib.request import urlopen as _urlopen
from typing import Dict, List


class scatterPlotSpatialPoints:
    """

            :Description:
                    scatterPlotSpatialPoints is used to convert the given data and plot the points.

            :param  iFile: str :
                    Name of the Input file
            :param  sep: str :
                    This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.


            **Importing this algorithm into a python program**
            --------------------------------------------------------
            .. code-block:: python

            from PAMI.extras.syntheticDataGenerator import scatterPlotSpatialPoints as plt

            obj = plt.scatterPlotSpatialPoints(iFile, "\t" )

            obj.save(oFile)


        """

    def __init__(self, iFile: str, sep: str = '\t') ->None:

        self._iFile = iFile
        self._sep = sep

    def _scanningPoints(self) -> Dict[str, str]:

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
            if self._iFile.startswith(('http:', 'https:')):
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

    def scatterPlotSpatialPoints(self) -> None:
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
