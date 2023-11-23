# DF2DB in this code dataframe is converting databases into sparse or dense transactional, temporal, Utility.
#
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.DF2DB import DF2DB as db
#
#     obj = db.DF2DB(idf, ">=", 16, "sparse/dense")
#
#     obj.getTransactional("outputFileName") # To create transactional database
#
#     obj.getTemporal("outputFileName") # To create temporal database
#
#     obj.getUtility("outputFileName") # To create utility database
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
from PAMI.extras.DF2DB.denseDF2DB import *
from PAMI.extras.DF2DB.sparseDF2DB import *
import sys

class DF2DB:
    """
        :Description:  This class will create database for given DataFrame based on Threshold values and conditions are defined in the class.
                       Converts Dataframe into sparse or dense dataframes.


        :param inputDF: DataFrame :
             It is sparse or dense DataFrame
        :param thresholdValue: int or float :
             It is threshold value of all item
        :param condition: str :
             It is condition of all item
        :param DFtype: str :
             It is DataFrame type. It should be sparse or dense. Default DF is sparse.


        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

                    from PAMI.extras.DF2DB import DF2DB as db

                    obj = db.DF2DB(idf, ">=", 16, "sparse/dense")

                    obj.getTransactional("outputFileName") # To create transactional database

                    obj.getTemporal("outputFileName") # To create temporal database

                    obj.getUtility("outputFileName") # To create utility database


        """


    def __init__(self, inputDF, thresholdValue, condition, DFtype='sparse') -> None:
        self.inputDF = inputDF
        self.thresholdValue = thresholdValue
        self.condition = condition
        self.DFtype = DFtype.lower()
        if DFtype == 'sparse':
            self.DF2DB = sparseDF2DB(self.inputDF, self.condition, self.thresholdValue)
        elif DFtype == 'dense':
            self.DF2DB = denseDF2DB(self.inputDF, self.condition, self.thresholdValue)
        else:
            raise Exception('DF type should be sparse or dense')

    def getTransactionalDatabase(self, outputFile) -> str:
        """
        create transactional database and return outputFileName

        :param outputFile: file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createTransactional(outputFile)
        return self.DF2DB.getFileName()

    def getTemporalDatabase(self, outputFile) -> str:
        """
        create temporal database and return outputFile name

        :param outputFile: file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createTemporal(outputFile)
        return self.DF2DB.getFileName()

    def getUtilityDatabase(self, outputFile) -> str:
        """
        create utility database and return outputFile name


        :param outputFile:  file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createUtility(outputFile)
        return self.DF2DB.getFileName()


if __name__ == '__main__':
    obj = DF2DB(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    obj.getTransactionalDatabase(sys.argv[5])