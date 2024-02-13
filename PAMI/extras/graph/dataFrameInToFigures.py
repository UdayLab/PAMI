# dataFrameInToFigures is used to convert the given dataframe into figures.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.graph import dataFrameInToFigures as fig
#
#     obj = fig.dataFrameInToFigures(idf)
#
#     obj.plotGraphsFromDataFrame("minSup", "patterns")
#
#     obj.plotGraphsFromDataFrame("minSup", "memory")
#
#     obj.plotGraphsFromDataFrame("minSup", "runtime")
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

import sys

import plotly.express as _px
import pandas as _pd

class dataFrameInToFigures():
    """

    :Description:   DataFrameInToFigures is used to convert the given dataframe into figures.

    :param  dataFrame:
            Name of the input dataframe
    :param xcolumn:
           Specify the name of the X-axis 
    :param ycolumn:
           Specify the name of the Y-axis 
           
    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.graph import dataframeInToFigures as fig

            obj = fig.dataframeInToFigures(idf)

            obj.plotGraphsFromDataFrame("minSup", "patterns")

            obj.plotGraphsFromDataFrame("minSup", "memory")

            obj.plotGraphsFromDataFrame("minSup", "runtime")

    """

    def __init__(self, dataFrame: _pd.DataFrame) -> None:
        self._dataFrame = dataFrame

    def plotGraphsFromDataFrame(self, xColumn, yColumn) -> None:
        """
        To plot graphs from given dataframe
        """
        fig = _px.line(self._dataFrame, x=self._dataFrame[xColumn] , y=self._dataFrame[yColumn], color=self._dataFrame.iloc[:, 0], labels={'x': xColumn, 'y': yColumn})
        fig.show()


if __name__ == '__main__':
    ab = dataFrameInToFigures(result)
    # user can change x and y columns
    ab.plotGraphsFromDataFrame("minSup", "patterns")


