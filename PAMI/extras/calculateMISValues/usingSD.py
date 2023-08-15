# usingSD is one of the fundamental algorithm to discover transactions in a database. It also stores the frequent patterns in the database.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.frequentPattern.basic import FPGrowth as alg
#
#     obj = alg.usingSD(iFile, minSup)
#
#     frequentPatterns = obj.getPatterns()
#
#     print("Total number of Frequent Patterns:", len(frequentPatterns))
#
#     obj.save(oFile)
#
#     Df = obj.getPatternInDataFrame()
#
#     memUSS = obj.getMemoryUSS()
#
#     print("Total Memory in USS:", memUSS)
#
#     memRSS = obj.getMemoryRSS()
#
#     print("Total Memory in RSS", memRSS)
#
#     run = obj.getRuntime()
#
#     print("Total ExecutionTime in seconds:", run)
#
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
import sys as _sys
import pandas as _pd
import validators as _validators
import statistics as _statistics
from urllib.request import urlopen as _urlopen

class usingBeta():
    """

            :Description: usingBeta is one of the fundamental algorithm to discover transactions in a database. It also stores the frequent patterns in the database.

            :param  iFile: str :
                           Name of the Input file to mine complete set of frequent patterns
            :param  oFile: str :
                           Name of the output file to store complete set of frequent patterns
            :param  minSup: int or float or str :
                           The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            :param  sep: str :
                           This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.



            :Attributes:

                startTime : float
                  To record the start time of the mining process

                endTime : float
                  To record the completion time of the mining process

                finalPatterns : dict
                  Storing the complete set of patterns in a dictionary variable

                memoryUSS : float
                  To store the total amount of USS memory consumed by the program

                memoryRSS : float
                  To store the total amount of RSS memory consumed by the program

                Database : list
                  To store the transactions of a database in list

                mapSupport : Dictionary
                    To maintain the information of item and their frequency

                finalPatterns : dict
                    it represents to store the patterns


            **Methods to execute code on terminal**
            --------------------------------------------------------
                Format:
                          >>> python3 usingSD.py <inputFile> <outputFile> <minSup>

                Example:
                          >>> python3 usingSD.py sampleDB.txt patterns.txt 10.0

                .. note:: minSup will be considered in percentage of database transactions


            **Importing this algorithm into a python program**
            --------------------------------------------------------
            .. code-block:: python

                        from PAMI.frequentPattern.basic import FPGrowth as alg

                        obj = alg.usingSD(iFile, minSup)

                        frequentPatterns = obj.getPatterns()

                        print("Total number of Frequent Patterns:", len(frequentPatterns))

                        obj.save(oFile)

                        Df = obj.getPatternInDataFrame()

                        memUSS = obj.getMemoryUSS()

                        print("Total Memory in USS:", memUSS)

                        memRSS = obj.getMemoryRSS()

                        print("Total Memory in RSS", memRSS)

                        run = obj.getRuntime()

                        print("Total ExecutionTime in seconds:", run)

                """

    _iFile: str = ' '
    _sd: int = int()
    _sep: str = str()
    _threshold: int = int()
    _finalPatterns: dict = {}
    __memoryUSS = float()
    __memoryRSS = float()
    __startTime = float()
    __endTime = float()
    _Database = []
    _mapSupport = {}

    def __init__(self, iFile: str, threshold: int, sep: str):
        self._iFile = iFile
        self._threshold = threshold
        self._sep = sep

    def _creatingItemSets(self) -> None:
        """
            Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        self._mapSupport = {}
        if isinstance(self._iFile, _pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()

        if isinstance(self._iFile, str):
            if _validators.url(self._iFile):
                data = _urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r') as f:
                        for line in f:
                            self._lno += 1
                            splitter = [i.rstrip() for i in line.split(self._sep)]
                            splitter = [x for x in splitter if x]
                            self._Database.append(splitter)
                except IOError:
                    print("File Not Found")

    def _creatingFrequentItems(self) -> tuple:
        """
        This function creates frequent items from _database.
        :return: frequentTidData that stores frequent items and their tid list.
        """
        tidData = {}
        self._lno = 0
        for transaction in self._Database:
            self._lno = self._lno + 1
            for item in transaction:
                if item not in tidData:
                    tidData[item] = [self._lno]
                else:
                    tidData[item].append(self._lno)
        mini = min([len(k) for k in tidData.values()])
        sd = _statistics.stdev([len(k) for k in tidData.values()])
        frequentTidData = {k: len(v) - sd for k, v in tidData.items()}
        return mini, frequentTidData

    def caculateMIS(self) -> None:
        self._creatingItemSets()
        mini, frequentItems = self._creatingFrequentItems()
        for x, y in frequentItems.items():
            if y < self._threshold:
                self._finalPatterns[x] = mini
            else:
                self._finalPatterns[x] = y

    def getPatternsAsDataFrame(self) -> _pd.DataFrame:
        """Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _pd.DataFrame(data, columns=['Items', 'MIS'])
        return dataFrame
    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process

        :rtype: float
        """

        return self.__memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process

        :rtype: float
        """

        return self.__memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process


        :return: returning total amount of runtime taken by the mining process

        :rtype: float
        """

        return self.__endTime - self.__startTime

    def save(self, outFile: str) -> None:
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)
    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """ this function is used to print the results
        """
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())

if __name__ == '__main__':
    cd = usingBeta("sample.txt", 10, ' ')
    cd.caculateMIS()
    cd.save('output.txt')