# DF2DB in this code dataframe is converting databases into sparse or dense transactional, temporal, Utility.
#
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#             from PAMI.extras.DF2DB import DF2DB as db
#
#             obj = db.DF2DB(idf, "sparse/dense")
#
#             obj.convert2Transactional("outputFileName", ">=", 16) # To create transactional database
#
#             obj.convert2Temporal("outputFileName", ">=", 16) # To create temporal database
#
#             obj.convert2Utility("outputFileName") # To create utility database
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
import PAMI.extras.convert.denseDF2DB as dense
import PAMI.extras.convert.sparseDF2DB as sparse
import sys,psutil,os,time
from typing import Union

class DF2DB:
    """
    :Description:  This class will create database for given DataFrame based on Threshold values and conditions are defined in the class.
                   Converts Dataframe into sparse or dense dataframes.

    :Attributes:

        :param inputDF: DataFrame :
             It is sparse or dense DataFrame
        :param thresholdValue: int or float :
             It is threshold value of all item
        :param condition: str :
             It is condition of all item
        :param DFtype: str :
             It is DataFrame type. It should be sparse or dense. Default DF is sparse.
        :param memoryUSS : float
            To store the total amount of USS memory consumed by the program
        :param memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        :param startTime : float
            To record the start time of the mining process
        endTime : float
            To record the completion time of the mining process


    **Importing this algorithm into a python program**
    --------------------------------------------------------
    .. code-block:: python

            from PAMI.extras.DF2DB import DF2DB as db

            obj = db.DF2DB(idf, "sparse/dense")

            obj.convert2Transactional("outputFileName",condition,threshold) # To create transactional database

            obj.convert2Temporal("outputFileName",condition,threshold) # To create temporal database

            obj.convert2Utility("outputFileName",condition,threshold) # To create utility database
    """,


    def __init__(self, inputDF, DFtype='dense') -> None:
        self.inputDF = inputDF
        self.DFtype = DFtype.lower()
        if DFtype == 'sparse':
            self.DF2DB = sparse.sparseDF2DB(self.inputDF)
        elif DFtype == 'dense':
            self.DF2DB = dense.denseDF2DB(self.inputDF)
        else:
            raise Exception('DF type should be sparse or dense')
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()

    def convert2TransactionalDatabase(self, oFile: str, condition: str, thresholdValue: Union[int, float]) -> str:
        """
        create transactional database and return oFileName
        :param oFile: file name or path to store database
        :type oFile: str
        :return: oFile name
        :rtype: str
        """
        self._startTime = time.time()
        self.DF2DB.convert2TransactionalDatabase(oFile,condition,thresholdValue)
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        self._endTime  = time.time()
        return self.DF2DB.getFileName()

    def convert2TemporalDatabase(self, oFile: str, condition: str, thresholdValue: Union[int, float]) -> str:
        """
        create temporal database and return oFile name
        :param oFile: file name or path to store database
        :type oFile: str
        :return: oFile name
        :rtype: str
        """
        self._startTime = time.time()
        self.DF2DB.convert2TemporalDatabase(oFile,condition,thresholdValue)
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        self._endTime  = time.time()
        return self.DF2DB.getFileName()

    def convert2UtilityDatabase(self, oFile: str) -> str:
        """
        create utility database and return oFile name
        :param oFile:  file name or path to store database
        :type oFile: str
        :return: outputFile name
        :rtype: str
        """
        self._startTime = time.time()
        self.DF2DB.convert2UtilityDatabase(oFile)
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        self._endTime = time.time()
        return self.DF2DB.getFileName()

    def convert2geoReferencedTransactionalDatabase(self, oFile: str, condition: str, thresholdValue: Union[int, float]) -> str:
        """
        create transactional database and return oFileName
        :param oFile: file name or path to store database
        :type oFile: str
        :return: oFile name
        :rtype: str
        """
        self._startTime = time.time()
        self.DF2DB.convert2TransactionalDatabase(oFile,condition,thresholdValue)
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        self._endTime  = time.time()
        return self.DF2DB.getFileName()

    def convert2geoReferencedTemporalDatabase(self, oFile: str, condition: str, thresholdValue: Union[int, float]) -> str:
        """
        create temporal database and return oFile name
        :param oFile: file name or path to store database
        :type oFile: str
        :return: oFile name
        :rtype: str
        """
        self._startTime = time.time()
        self.DF2DB.convert2TemporalDatabase(oFile,condition,thresholdValue)
        process = psutil.Process(os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        self._endTime  = time.time()
        return self.DF2DB.getFileName()

    def getMemoryUSS(self) -> float:
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self) -> float:
        """
        Calculating the total amount of runtime taken by the mining process


        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self._endTime - self._startTime



if __name__ == '__main__':
    obj = DF2DB(sys.argv[1])
    obj.getTransactionalDatabase(sys.argv[2],sys.argv[3],sys.argv[4])
    print("Conversion is complete.")
    print("Total Memory in USS:", obj.getMemoryUSS())
    print("Total Memory in RSS", obj.getMemoryRSS())
    print("Total ExecutionTime in ms:", obj.getRuntime())
