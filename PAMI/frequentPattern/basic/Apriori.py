# Apriori is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs apriori property (or downward closure property) to  reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a transactional database.
#
# **Importing this algorithm into a python program**
#
#             import PAMI.frequentPattern.basic.Apriori as alg
#
#             iFile = 'sampleDB.txt'
#
#             minSup = 10  # can also be specified between 0 and 1
#
#             obj = alg.Apriori(iFile, minSup)
#
#             obj.mine()
#
#             frequentPattern = obj.getPatterns()
#
#             print("Total number of Frequent Patterns:", len(frequentPattern))
#
#             obj.save(oFile)
#
#             Df = obj.getPatternInDataFrame()
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime()
#
#             print("Total ExecutionTime in seconds:", run)
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

from PAMI.frequentPattern.basic import abstract as _ab
from typing import Dict, Union
from deprecated import deprecated


class Apriori(_ab._frequentPatterns):
    """
    **About this algorithm**

    :**Description**: Apriori is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs apriori property (or downward closure property) to  reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a transactional database.

    :**Reference**:  Agrawal, R., Imieli ́nski, T., Swami, A.: Mining association rules between sets of items in large databases.
                     In: SIGMOD. pp. 207–216 (1993), https://doi.org/10.1145/170035.170072

    :**Parameters**:    - **iFile** (*str or URL or dataFrame*) -- *Name of the Input file to mine complete set of frequent patterns.*
                        - **oFile** (*str*) -- *Name of the output file to store complete set of frequent patterns.*
                        - **minSup** (*int or float or str*) -- *The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.*
                        - **sep** (*str*) -- *This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the mining process.*
                        - **endTime** (*float*) -- *To record the completion time of the mining process.*
                        - **finalPatterns** (*dict*) -- *Storing the complete set of patterns in a dictionary variable.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*
                        - **Database** (*list*) -- *To store the transactions of a database in list.*


    **Execution methods**

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 Apriori.py <inputFile> <outputFile> <minSup>

      Example Usage:

      (.venv) $ python3 Apriori.py sampleDB.txt patterns.txt 10.0

    .. note:: minSup can be specified  in support count or a value between 0 and 1.


    **Calling from a python program**

    .. code-block:: python

            import PAMI.frequentPattern.basic.Apriori as alg

            iFile = 'sampleDB.txt'

            minSup = 10  # can also be specified between 0 and 1

            obj = alg.Apriori(iFile, minSup)

            obj.mine()

            frequentPattern = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPattern))

            obj.save(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


    **Credits**

    The complete program was written by P. Likhitha  and revised by Tarun Sreepada under the supervision of Professor Rage Uday Kiran.

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
            #temp = []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
                self._Database = [x.split(self._sep) for x in self._Database]
            else:
                print("The column name should be Transactions and each line should be separated by tab space or a seperator specified by the user")
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
        :type value: int or float or str
        :return: converted type
        :rtype: int or float
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

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """
        Frequent pattern mining process will start from here
        """
        self.mine()

    def mine(self, memorySaver = True) -> None:
        """
        Frequent pattern mining process will start from here

        Attributes
        ----------
        memorySaver : bool
            This attribute is used to enable or disable memory saving mode. By default, it is enabled.
            It saves the memory by deleting the intermediate results after the completion of the mining process.
        """
        self._Database = []
        self._startTime = _ab._time.time()

        self._creatingItemSets()

        self._minSup = self._convert(self._minSup)

        items = {}
        index = 0
        for line in self._Database:
            for item in line:
                if tuple([item]) in items:
                    items[tuple([item])].append(index)
                else:
                    items[tuple([item])] = [index]
            index += 1

        # sort by length in descending order
        items = dict(sorted(items.items(), key=lambda x: len(x[1]), reverse=True))

        cands = []
        fileData = {}
        for key in items:
            if len(items[key]) >= self._minSup:
                cands.append(key)
                # self._finalPatterns["\t".join(key)] = len(items[key])
                self._finalPatterns[key] = len(items[key])
                fileData[key] = set(items[key])
            else:
                break

        if memorySaver:
            while cands:
                newKeys = []
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        if cands[i][:-1] == cands[j][:-1]:
                            newCand = cands[i] + tuple([cands[j][-1]])
                            intersection = fileData[tuple([newCand[0]])]
                            for k in range(1, len(newCand)):
                                intersection = intersection.intersection(fileData[tuple([newCand[k]])])
                            if len(intersection) >= self._minSup:
                                newKeys.append(newCand)
                                self._finalPatterns[newCand] = len(intersection)
                del cands
                cands = newKeys
                del newKeys
        else:
            while cands:
                newKeys = []
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        if cands[i][:-1] == cands[j][:-1]:
                            newCand = cands[i] + tuple([cands[j][-1]])
                            intersection = fileData[cands[i]] & fileData[cands[j]]
                            # intersection = fileData[tuple([newCand[0]])]
                            # for k in range(1, len(newCand)):
                                # intersection = intersection.intersection(fileData[tuple([newCand[k]])])
                            if len(intersection) >= self._minSup:
                                newKeys.append(newCand)
                                self._finalPatterns[newCand] = len(intersection)
                                fileData[newCand] = intersection
                del cands
                cands = newKeys
                del newKeys
        

        process = _ab._psutil.Process(_ab._os.getpid())
        self._endTime = _ab._time.time()
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Apriori algorithm ")

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

    def getPatternsAsDataFrame(self) -> _ab._pd.DataFrame:
        """

        Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame

        """

        # time = _ab._time.time()
        # dataFrame = {}
        # data = []
        # for a, b in self._finalPatterns.items():
        #     # data.append([a.replace('\t', ' '), b])
        #     data.append([" ".join(a), b])
        #     dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        # print("Time taken to convert the frequent patterns into DataFrame is: ", _ab._time.time() - time)

        dataFrame = _ab._pd.DataFrame(list([[" ".join(x), y] for x,y in self._finalPatterns.items()]), columns=['Patterns', 'Support'])

        return dataFrame

    def save(self, oFile: str, seperator = "\t" ) -> None:
        """

        Complete set of frequent patterns will be loaded in to an output file

        :param oFile: name of the output file
        :type oFile: csvfile
        :param seperator: variable to store separator value
        :type seperator: string
        :return: None
        """

        # self._oFile = oFile
        # writer = open(self._oFile, 'w+')
        # for x, y in self._finalPatterns.items():
        #     patternsAndSupport = x.strip() + ":" + str(y[0])
        #     writer.write("%s \n" % patternsAndSupport)
        with open(oFile, 'w') as f:
            for x, y in self._finalPatterns.items():
                x = seperator.join(x)
                f.write(f"{x}:{y}\n")

    def getPatterns(self) -> Dict[str, int]:
        """

        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self) -> None:
        """
        This function is used to print the result
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
        _ap.mine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ap._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

