# DF2DBPlus in this code the dense dataframe is converting databases into different transactional, temporal, utility types.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.DF2DB import DF2DBPlus as dfdbp
#
#     obj = dfdbp.DF2DBPlus(idf, ">=", 16)
#
#     obj.getTransactional("outputFileName") # To create a transactional database
#
#     obj.getTDB("outputFileName")   # To create a temporal database
#
#     obj.getUDB("outputFileName")    # To create a utility database
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

from PAMI.extras.DF2DB.denseDF2DBPlus import *
from PAMI.extras.DF2DB.sparseDF2DBPlus import *
import sys


class DF2DBPlus:
    """
            :Description:  This class create database from DataFrame. Threshold values and conditions are defined to each item by thresholdConditonDF.

            :param inputDF: DataFrame :
                 It is sparse or dense DataFrame
            :param thresholdConditionDF: pandas.DataFrame :
                It is DataFrame to contain threshold values and condition each item
            :param condition: str :
                 It is condition of all item
            :param DFtype: str :
                 It is DataFrame type. It should be sparse or dense. Default DF is sparse.



        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

                     from PAMI.extras.DF2DB import DF2DBPlus as dfdbp

                     obj = dfdbp.DF2DBPlus(idf, ">=", 16)

                     obj.getTransactional("outputFileName") # To create a transactional database

                     obj.getTDB("outputFileName")   # To create a temporal database

                     obj.getUDB("outputFileName")    # To craete a utility database

    """

    def __init__(self, inputDF, thresholdConditionDF, DFtype='sparse') -> None:
        self.inputDF = inputDF
        self.thresholdConditionDF = thresholdConditionDF
        self.DFtype = DFtype.lower()
        if DFtype == 'sparse':
            self.DF2DB = sparseDF2DBPlus(self.inputDF, self.thresholdConditionDF)
        elif DFtype == 'dense':
            self.DF2DB = denseDF2DBPlus(self.inputDF, self.thresholdConditionDF)
        else:
            raise Exception('DF type should be sparse or dense')

    def getTransactional(self, outputFile) -> str:
        """
        create transactional database and return outputFileName

        :param outputFile: file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createTransactional(outputFile)
        return self.DF2DB.getFileName()

    def getTDB(self, outputFile) -> str:
        """
        create temporal database and return outputFile name

        :param outputFile: file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createTemporal(outputFile)
        return self.DF2DB.getFileName()

    def getUDB(self, outputFile) -> str:
        """
        create utility database and return outputFile name

        :param outputFile:  file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createUtility(outputFile)
        return self.DF2DB.getFileName()
if __name__ == '__main__':
    obj = DF2DBPlus(sys.argv[1], sys.argv[2], sys.argv[3])
    obj.getTransactional(sys.argv[4])