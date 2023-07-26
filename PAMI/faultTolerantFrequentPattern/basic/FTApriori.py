# FTApriori is one of the fundamental algorithm to discover fault-tolerant frequent patterns in a transactional database.
#
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------------------
#
#     from PAMI.faultTolerantFrequentPattern.basic import FTApriori as alg
#
#     obj = alg.FTApriori(inputFile,minSup,itemSup,minLength,faultTolerance)
#
#     obj.startMine()
#
#     patterns = obj.getPatterns()
#
#     print("Total number of fault-tolerant frequent patterns:", len(patterns))
#
#     obj.save("outputFile")
#
#     memUSS = obj.getMemoryUSS()
#
#     print("Total Memory in USS:", memUSS)
#
#     memRSS = obj.getMemoryRSS()
#
#     print("Total Memory in RSS", memRSS)
#
#     run = obj.getRuntime
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

from PAMI.faultTolerantFrequentPattern.basic import abstract as _ab


class FTApriori(_ab._faultTolerantFrequentPatterns):
    """
    
    :Description:   FT-Apriori is one of the fundamental algorithm to discover fault-tolerant frequent patterns in a transactional database.
                    This program employs apriori property (or downward closure property) to  reduce the search space effectively.

    :Reference:       Pei, Jian & Tung, Anthony & Han, Jiawei. (2001). Fault-Tolerant Frequent Pattern Mining: Problems and Challenges.

    :param  iFile: str :
           Name of the Input file to mine complete set of frequent patterns

    :param  oFile: str :
                   Name of the output file to store complete set of frequent patterns

    :param  minSup: float or int or str :
                    The user can specify minSup either in count or proportion of database size.
                    If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                    Otherwise, it will be treated as float.
                    Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float

    :param  itemSup: int or float :
                    Frequency of an item

    :param minLength: int :
                    minimum length of a pattern

    :param faultTolerance: int

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
    ---------------------------------------
    
            Format:
                      >>>    python3 FTApriori.py <inputFile> <outputFile> <minSup> <itemSup> <minLength> <faultTolerance>
            Example:
                      >>>    python3 FTApriori.py sampleDB.txt patterns.txt 10.0 3.0 3 1
    
            .. note:: minSup will be considered in times of minSup and count of database transactions
    
    **Importing this algorithm into a python program**
    ----------------------------------------------------------------
    .. code-block:: python
    
            from PAMI.faultTolerantFrequentPattern.basic import FTApriori as alg
    
            obj = alg.FTApriori(inputFile,minSup,itemSup,minLength,faultTolerance)
    
            obj.startMine()
    
            patterns = obj.getPatterns()
    
            print("Total number of fault-tolerant frequent patterns:",  len(patterns))
    
            obj.savePatterns("outputFile")
    
            memUSS = obj.getMemoryUSS()
    
            print("Total Memory in USS:",  memUSS)
    
            memRSS = obj.getMemoryRSS()
    
            print("Total Memory in RSS",  memRSS)
    
            run = obj.getRuntime
    
            print("Total ExecutionTime in seconds:",  run)
    
    **Credits:**
    ----------------
             The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.

        """

    _minSup = float()
    _itemSup = float()
    _minLength = int()
    _faultTolerance = int()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _mapSupport = {}

    def _creatingItemSets(self):
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

    def _convert(self, value):
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

    def _Count(self, k):
        """
        param k: list of items
        type k: list
        """
        count = 0
        items = []
        k = list(k)
        n = len(k) - self._faultTolerance
        c = _ab._itertools.combinations(k, n)
        count = 0
        for j in c:
            j = list(j)
            for i in self._Database:
                if set(j).issubset(i):
                    count += 1
                    items.append(i)
        items = list(set(map(tuple, items)))
        return len(items), items

    def _oneLengthFrequentItems(self):
        self._mapSupport = {}
        for li in self._Database:
            for i in li:
                if i not in self._mapSupport:
                    self._mapSupport[i] = 1
                else:
                    self._mapSupport[i] += 1
        self._mapSupport = {k: v for k, v in self._mapSupport.items() if v >= self._itemSup}

    def _countItemSupport(self, itemset):
        tids = {}
        res = True
        count = 0
        for x in self._Database:
            if abs(len(itemset) - len(set(x) & set(itemset))) <= self._faultTolerance:
                count += 1
        return count

    def _getFaultPatterns(self):
        l = [k for k, v in self._mapSupport.items()]
        for i in range(0, len(l) + 1):
            c = _ab._itertools.combinations(l, i)
            for j in c:
                res = self._countItemSupport(j)
                if len(j) >= self._minLength and res >= self._minSup:
                    self._finalPatterns[tuple(j)] = res

    def startMine(self):
        """
            Fault-tolerant frequent pattern mining process will start from here
        """
        self._Database = []
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._itemSup = self._convert(self._itemSup)
        self._minLength = int(self._minLength)
        self._faultTolerance = int(self._faultTolerance)
        self._oneLengthFrequentItems()

        self._getFaultPatterns()
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Fault-Tolerant Frequent patterns were generated successfully using FTApriori algorithm ")

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process

        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            s = str()
            for i in a:
                s = s + i + ' '
            data.append([s, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to an output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s = str()
            for i in x:
                s = s + i + '\t'
            s1 = s.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """ this is function is used to print the result
        """
        print("Total number of Fault-Tolerant Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 7 or len(_ab._sys.argv) == 8:
        if len(_ab._sys.argv) == 8:
            _ap = FTApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4],
                            _ab._sys.argv[5], _ab._sys.argv[6], _ab._sys.argv[7], )
        if len(_ab._sys.argv) == 7:
            _ap = FTApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])
        _ap.startMine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
