# **Importing this algorithm into a python program**
# --------------------------------------------------------
#

#             import PAMI.periodicFrequentPattern.kPFPMiner as alg
#
#             obj = alg.kPFPMiner(iFile, k)
#
#             obj.startMine()
#
#             periodicFrequentPatterns = obj.getPatterns()
#
#             print("Total number of top-k Periodic Frequent Patterns:", len(periodicFrequentPatterns))
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
     Copyright (C)  2021 Rage Uday Kiran

"""

from PAMI.periodicFrequentPattern.basic import abstract as _ab
import pandas as pd
from deprecated import deprecated

from PAMI.periodicFrequentPattern.topk.kPFPMiner import abstract as _ab


class kPFPMiner(_ab._periodicFrequentPatterns):
    """
    :Description:   Top - K is and algorithm to discover top periodic-frequent patterns in a temporal database.

    :Reference:   Likhitha, P., Ravikumar, P., Kiran, R.U., Watanobe, Y. (2022).
                  Discovering Top-k Periodic-Frequent Patterns in Very Large Temporal Databases. Big Data Analytics.
                 BDA 2022. Lecture Notes in Computer Science, vol 13773. Springer, Cham. https://doi.org/10.1007/978-3-031-24094-2_14

    :param  iFile: str :
                   Name of the Input file to mine complete set of periodic frequent pattern's
    :param  oFile: str :
                   Name of the output file to store complete set of periodic frequent pattern's

    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.

    :Attributes:

        iFile : str
            Input file name or path of the input file
        k: int
            User specified counte of top-k periodic frequent patterns
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        oFile : str
            Name of the output file or the path of the output file
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program

    :Methods:

        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingItemSets()
            Scans the dataset or dataframes and stores in list format
        frequentOneItem()
            Generates one frequent patterns
        eclatGeneration(candidateList)
            It will generate the combinations of frequent items
        generateFrequentPatterns(tidList)
            It will generate the combinations of frequent items from a list of items

    **Executing the code on terminal:**
    ------------------------------------------
    .. code-block:: console


       Format:


       (.venv) $ python3 kPFPMiner.py <inputFile> <outputFile> <k>

       Examples :

       (.venv) $  python3 kPFPMiner.py sampleDB.txt patterns.txt 10


    **Sample run of the importing code:
    --------------------------------------
    .. code-block:: python

            import PAMI.periodicFrequentPattern.kPFPMiner as alg

            obj = alg.kPFPMiner(iFile, k)

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of top-k Periodic Frequent Patterns:", len(periodicFrequentPatterns))

            obj.save(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

    **Credits:**
    --------------
            The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

    """

    _startTime = float()
    _endTime = float()
    _k = int()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _tidList = {}
    lno = int()
    _maximum = int()

    def _creatingItemSets(self):
        """
        Storing the complete transactions of the database/input file in a database variable
        """

        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()

            # print(self.Database)
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()
                    
    def getPer_Sup(self, tids):
        tids.sort()
        cur=0
        per=list()
        sup=0
        #print(tids)
        for i in range(len(tids)-1):
            j = i + 1
            #if tids[j] - cur <= periodicity:
                #return [0,0]
            per.append(tids[j] - cur)
            cur = tids[j]
        per.append(self.lno - cur)
        return max(per)

    def _frequentOneItem(self):
        """
        Generating one frequent patterns
        """
        self._mapSupport = {}
        self._tidList = {}
        n = 0
        for line in self._Database:
            self.lno += 1
            n = int(line[0])
            for i in range(1, len(line)):
                si = line[i]
                if self._mapSupport.get(si) is None:
                    self._mapSupport[si] = [1, abs(0 - n), n]
                    self._tidList[si] = [n]
                else:
                    self._mapSupport[si][0] += 1
                    self._mapSupport[si][1] = max(self._mapSupport[si][1], abs(n - self._mapSupport[si][2]))
                    self._mapSupport[si][2] = n
                    self._tidList[si].append(n)
        for x, y in self._mapSupport.items():
            self._mapSupport[x][1] = max(self._mapSupport[x][1], abs(n - self._mapSupport[x][2]))
        plist = [key for key, value in sorted(self._mapSupport.items(), key=lambda x: x[1], reverse=True)]
        for i in plist:
            if len(self._finalPatterns) >= self._k:
                break
            else:
                self._finalPatterns[i] = self._mapSupport[i][1]
        self._maximum = max([self._finalPatterns[i] for i in self._finalPatterns.keys()])
        plist = list(self._finalPatterns.keys())
        return plist


    def _save(self, prefix, suffix, tidSetI):
        """Saves the patterns that satisfy the periodic frequent property.

        :param prefix: the prefix of a pattern
        :type prefix: list
        :param suffix: the suffix of a patterns
        :type suffix: list
        :param tidSetI: the timestamp of a patterns
        :type tidSetI: list
        """

        if prefix is None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        val = self.getPer_Sup(tidSetI)
        sample = str()
        for i in prefix:
            sample = sample + i + " "
        if len(self._finalPatterns) < self._k:
            if val < self._maximum:
                self._finalPatterns[sample] = val
                self._finalPatterns = {k: v for k, v in sorted(self._finalPatterns.items(), key=lambda item: item[1], reverse=True)}
                self._maximum = max([i for i in self._finalPatterns.values()])
        else:
            for x, y in sorted(self._finalPatterns.items(), key=lambda x: x[1], reverse=True):
                if val < y:
                    del self._finalPatterns[x]
                    self._finalPatterns[sample] = val
                    self._finalPatterns = {k: v for k, v in
                                              sorted(self._finalPatterns.items(), key=lambda item: item[1],
                                                     reverse=True)}
                    self._maximum = max([i for i in self._finalPatterns.values()])
                    return

    def _Generation(self, prefix, itemSets, tidSets):
        """Equivalence class is followed  and checks for the patterns generated for periodic-frequent patterns.

        :param prefix:  main equivalence prefix
        :type prefix: periodic-frequent item or pattern
        :param itemSets: patterns which are items combined with prefix and satisfying the periodicity and frequent with their timestamps
        :type itemSets: list
        :param tidSets: timestamps of the items in the argument itemSets
        :type tidSets: list

        """
        if len(itemSets) == 1:
            i = itemSets[0]
            tidI = tidSets[0]
            self._save(prefix, [i], tidI)
            return
        for i in range(len(itemSets)):
            itemI = itemSets[i]
            if itemI is None:
                continue
            tidSetI = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetX = [itemI]
            for j in range(i + 1, len(itemSets)):
                itemJ = itemSets[j]
                tidSetJ = tidSets[j]
                y = list(set(tidSetI).intersection(tidSetJ))
                if self.getPer_Sup(y) <= self._maximum:
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newPrefix = list(set(itemSetX)) + prefix
            self._Generation(newPrefix, classItemSets, classTidSets)
            self._save(prefix, list(set(itemSetX)), tidSetI)

    def _convert(self, value):
        """
        to convert the type of user specified minSup value

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
                value = ((len(self._Database)) * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
        Main function of the program

        """
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._k is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        self._k = self._convert(self._k)
        plist = self._frequentOneItem()
        for i in range(len(plist)):
            itemI = plist[i]
            tidSetI = self._tidList[itemI]
            itemSetX = [itemI]
            itemSets = []
            tidSets = []
            for j in range(i + 1, len(plist)):
                itemJ = plist[j]
                tidSetJ = self._tidList[itemJ]
                y1 = list(set(tidSetI).intersection(tidSetJ))
                if self.getPer_Sup(y1) <= self._maximum:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self._Generation(itemSetX, itemSets, tidSets)
        print("kPFPMiner has successfully generated top-k frequent patterns")
        self._endTime = _ab._time.time()
        self._memoryUSS = float()
        self._memoryRSS = float()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss

    def mine(self):
        """
        Main function of the program

        """
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._k is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        self._k = self._convert(self._k)
        plist = self._frequentOneItem()
        for i in range(len(plist)):
            itemI = plist[i]
            tidSetI = self._tidList[itemI]
            itemSetX = [itemI]
            itemSets = []
            tidSets = []
            for j in range(i + 1, len(plist)):
                itemJ = plist[j]
                tidSetJ = self._tidList[itemJ]
                y1 = list(set(tidSetI).intersection(tidSetJ))
                if self.getPer_Sup(y1) <= self._maximum:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self._Generation(itemSetX, itemSets, tidSets)
        print("kPFPMiner has successfully generated top-k frequent patterns")
        self._endTime = _ab._time.time()
        self._memoryUSS = float()
        self._memoryRSS = float()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
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
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'periodicity'])
        return dataFrame

    def save(self, outFile):
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
        print("Total number of  Top-k Periodic Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = kPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = kPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of top-k periodic frequent patterns:", len(_Patterns))
        _ap.save(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")


