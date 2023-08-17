# plotLineGraphFromDataFrame is used to convert the given dataframe into plotLineGraph.
#
#  **Importing this algorithm into a python program**
#  --------------------------------------------------------
#
#     from PAMI.extras.graph import plotLineGraphsFromDataFrame as plt
#
#     obj = plt.plotLineGraphsFromDictionary(idf)
#
#     obj.save()
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

import matplotlib.pyplot as plt
import pandas as _pd
import sys

class plotGraphsFromDataFrame():
    """
        plotLineGraphFromDataFrame is used to convert the given dataframe into plotLineGraph.

        Attributes:
        ----------
        :param dataFrame : DataFrame
            store input data as DataFrame

        Methods:
        -------
        plotLineGraphFromDatFrame()
            draw line graph of input data. input data's key is x and value is y.

            **Importing this algorithm into a python program**
            --------------------------------------------------------
            .. code-block:: python

            from PAMI.extras.graph import plotLineGraphsFromDataframe as plt

            obj = plt.plotLineGraphsFromDataFrame(idf)

            obj.save()
        """

    def __init__(self, dataFrame: _pd.DataFrame) -> None:

        self._dataFrame = dataFrame

    def plotGraphsFromDataFrame(self) -> None:
        self._dataFrame.plot(x='minSup', y='patterns', kind='line')
        plt.show()
        print('Graph for No Of Patterns is successfully generated!')
        self._dataFrame.plot(x='minSup', y='runtime', kind='line')
        plt.show()
        print('Graph for Runtime taken is successfully generated!')
        self._dataFrame.plot(x='minSup', y='memory', kind='line')
        plt.show()
        print('Graph for memory consumption is successfully generated!')




if __name__ == '__main__':
    #data = {'algorithm': ['FPGrowth','FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT'],
    #        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05],
    #        'patterns': [386, 155, 60, 36, 10, 386, 155, 60, 26, 10],
    #        'runtime': [7.351629, 4.658654 , 4.658654 , 1.946843, 1.909376, 4.574833, 2.514252, 1.834948, 1.889892, 1.809999],
    #        'memory': [426545152, 309182464, 241397760, 225533952, 220950528, 233537536, 267165696, 252841984, 245690368,
    #                    295710720]
    #        }
    '''data = {
        'algorithm': ['FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth'],
        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05],
        'patterns': [386, 155, 60, 36, 10],
        'runtime': [7.351629, 4.658654, 4.658654, 1.946843, 1.909376],
        'memory': [426545152, 309182464, 241397760, 225533952, 220950528]
        }'''
    dataFrame = _pd.DataFrame(data)
    ab = plotGraphsFromDataFrame(dataFrame)
    ab.plotGraphsFromDataFrame()
    obj = plotGraphsFromDataFrame(sys.argv[1])
    obj.plotGraphsFromDataFrame(sys.argv[2])