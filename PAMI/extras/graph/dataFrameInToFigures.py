# dataFrameInToFigures is a code used to convert the dataframe into Figures.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.graph import dataFrameInToFigures as fig
#
#     obj = fig.dataFrameInToFigures(idf )
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
import plotly.express as _px
import pandas as _pd

class dataFrameInToFigures():
    """

        :Description:
                dataFrameInToFigures is a code used to convert the dataframe into Figures

        :param  dataFrame: int or float :
                Name of the Input dataFrame.



        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

        from PAMI.extras.graph import dataFrameInToFigures as fig

        obj = fig.dataFrameInToFigures(idf )


        obj.save(oFile)


    """

    def __init__(self, dataFrame: _pd.DataFrame) -> None:

        self._dataFrame = dataFrame

    def plotGraphsFromDataFrame(self,xColumn: str='minSup',yColumn: str='patterns',lineLabels: str='algorithm') -> None:
        fig = _px.line(self._dataFrame, x=xColumn, y=yColumn, color=lineLabels)
        fig.show()






if __name__ == '__main__':
    #data = {'algorithm': ['FPGrowth','FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT', 'ECLAT'],
    #        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05],
    #        'patterns': [386, 155, 60, 36, 10, 386, 155, 60, 26, 10],
    #        'runtime': [7.351629, 4.658654 , 4.658654 , 1.946843, 1.909376, 4.574833, 2.514252, 1.834948, 1.889892, 1.809999],
    #        'memory': [426545152, 309182464, 241397760, 225533952, 220950528, 233537536, 267165696, 252841984, 245690368,
    #                    295710720]
    #        }
    data = {
        'algorithm': ['FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth', 'FPGrowth'],
        'minSup': [0.01, 0.02, 0.03, 0.04, 0.05],
        'patterns': [386, 155, 60, 36, 10],
        'runtime': [7.351629, 4.658654, 4.658654, 1.946843, 1.909376],
        'memory': [426545152, 309182464, 241397760, 225533952, 220950528]
        }
    dataFrame = _pd.DataFrame(data)
    ab = dataFrameInToFigures(dataFrame)
    ab.plotGraphsFromDataFrame()