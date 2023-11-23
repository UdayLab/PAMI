# Apriori is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs apriori property (or downward closure property) to  reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a transactional database.
#
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------
#
#     import PAMI.frequentPattern.basic.Apriori as alg
#
#     obj = alg.Apriori(iFile, minSup)
#
#     obj.startMine()
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

from PAMI.frequentPattern.basic import abstract as _ab
from typing import List, Dict, Tuple, Set, Union, Any, Generator

class Apriori(_ab._frequentPatterns):
    """
    :Description: Apriori is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs apriori property (or downward closure property) to  reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a transactional database.

    :Reference:  Agrawal, R., Imieli ́nski, T., Swami, A.: Mining association rules between sets of items in large databases.
            In: SIGMOD. pp. 207–216 (1993), https://doi.org/10.1145/170035.170072

    :param  iFile: str :
                   Name of the Input file to mine complete set of frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of frequent patterns
    :param  minSup: int :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
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



    **Methods to execute code on terminal**
    ----------------------------------------------------

            Format:
                      >>> python3 Apriori.py <inputFile> <outputFile> <minSup>

            Example:
                      >>>  python3 Apriori.py sampleDB.txt patterns.txt 10.0

            .. note:: minSup will be considered in percentage of database transactions


    **Importing this algorithm into a python program**
    ----------------------------------------------------

    .. code-block:: python

             import PAMI.frequentPattern.basic.Apriori as alg

             obj = alg.Apriori(iFile, minSup)

             obj.startMine()

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


    **Credits:**
    -------------

             The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

    """

    _minSup = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []

    def _creatingItemSets(self) -> None:
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            temp = []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                temp = self._iFile['Transactions'].tolist()

            for k in temp:
                self._Database.append(set(k))
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(set(temp))
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(set(temp))
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value: Union[int, float, str]) -> Union[int, float]:
        """
        To convert the user specified minSup value

        :param value: user specified minSup value

        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self._Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self._Database) * value)
            else:
                value = int(value)
        return value

    def _candidateToFrequent(self, candidateList: List[set]) -> Dict[frozenset, int]:
        """Generates frequent patterns from the candidate patterns

        :param candidateList: Candidate patterns will be given as input

        :type candidateList: list

        :return: returning set of all frequent patterns

        :rtype: dict
        """

        candidateToFrequentList = {}
        for i in self._Database:
            dictionary = {frozenset(j): int(candidateToFrequentList.get(frozenset(j), 0)) + 1 for j in candidateList if
                          j.issubset(i)}
            candidateToFrequentList.update(dictionary)
        candidateToFrequentList = {key: value for key, value in candidateToFrequentList.items() if
                                   value >= self._minSup}

        return candidateToFrequentList

    @staticmethod
    def _frequentToCandidate(frequentList: Dict[frozenset, int], length: int) -> List[set]:
        """Generates candidate patterns from the frequent patterns

        :param frequentList: set of all frequent patterns to generate candidate patterns of each of size is length

        :type frequentList: dict

        :param length: size of each candidate patterns to be generated

        :type length: int

        :return: set of candidate patterns in sorted order

        :rtype: list
        """

        frequentToCandidateList = []
        for i in frequentList:
            nextList = [i | j for j in frequentList if len(i | j) == length and (i | j) not in frequentToCandidateList]
            frequentToCandidateList.extend(nextList)
        return sorted(frequentToCandidateList)

    def startMine(self) -> None:
        """
            Frequent pattern mining process will start from here
        """
        self._Database = []
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        itemsList = sorted(list(set.union(*self._Database)))  # because Database is list
        items = [{i} for i in itemsList]
        itemsCount = len(items)
        self._minSup = self._convert(self._minSup)
        self._finalPatterns = {}
        for i in range(1, itemsCount):
            frequentSet = self._candidateToFrequent(items)
            for x, y in frequentSet.items():
                sample = str()
                for k in x:
                    sample = sample + k + "\t"
                self._finalPatterns[sample] = y
            items = self._frequentToCandidate(frequentSet, i + 1)
            if len(items) == 0:
                break  # finish apriori
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Apriori algorithm ")

    def getMemoryUSS(self) -> float:
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self) -> float:
        """Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process

        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self) -> _ab._pd.DataFrame:
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        # dataFrame = dataFrame.replace(r'\r+|\n+|\t+',' ', regex=True)
        return dataFrame

    def save(self, outFile) -> None:
        """Complete set of frequent patterns will be loaded in to an output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self) -> Dict[str, int]:
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self) -> None:
        """
        this function is used to print the result

        """
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = Apriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = Apriori(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ap._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

