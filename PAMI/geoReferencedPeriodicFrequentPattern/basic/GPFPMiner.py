# GPFPMiner is a Extension of ECLAT algorithm,which  stands for Equivalence Class Clustering and bottom-up
# Lattice Traversal to mine the geo referenced peridoic frequent patterns.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#
#     import PAMI.geoReferencedPeridicFrequentPattern.GPFPMiner as alg
#
#     obj = alg.GPFPMiner("sampleTDB.txt", "sampleN.txt", 5, 3)
#
#     obj.startMine()
#
#     Patterns = obj.getPatterns()
#
#     print("Total number of Geo Referenced Periodic-Frequent Patterns:", len(Patterns))
#
#     obj.save("outFile")
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

from  PAMI.geoReferencedPeriodicFrequentPattern.basic import abstract as _ab


class GPFPMiner(_ab._geoReferencedPeriodicFrequentPatterns):
    """ 
    Description:
    ------------
        GPFPMiner is a Extension of ECLAT algorithm,which  stands for Equivalence Class Clustering and bottom-up
        Lattice Traversal to mine the geo referenced peridoic frequent patterns.
        
    Reference:
    -----------
    
          
    Attributes :
    ----------
            iFile : str
                Input file name or path of the input file
            nFile: str:
               Name of Neighbourhood file name
            minSup: float or int or str
                The user can specify minSup either in count or proportion of database size.
                If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                Otherwise, it will be treated as float.
                Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
            maxPer: float or int or str
                The user can specify maxPer either in count or proportion of database size.
                If the program detects the data type of maxPer is integer, then it treats minSup is expressed in count.
                Otherwise, it will be treated as float.
                Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
            sep : str
                This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
                However, the users can override their default separator.
            startTime:float
                To record the start time of the mining process
            endTime:float
                To record the completion time of the mining process
            finalPatterns: dict
                Storing the complete set of patterns in a dictionary variable
            oFile : str
                Name of the output file to store complete set of frequent patterns
            memoryUSS : float
                To store the total amount of USS memory consumed by the program
            memoryRSS : float
                To store the total amount of RSS memory consumed by the program
            Database : list
                To store the complete set of transactions available in the input database/file
    Methods :
    ---------
            startMine()
                Mining process will start from here
            getPatterns()
                Complete set of patterns will be retrieved with this function
            save(oFile)
                Complete set of frequent patterns will be loaded in to a output file
            getPatternsAsDataFrames()
                Complete set of frequent patterns will be loaded in to a dataframe
            getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
            getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
            getRuntime()
                Total amount of runtime taken by the mining process will be retrieved from this function
            creatingItemSets(iFileName)
                Storing the complete transactions of the database/input file in a database variable
            frequentOneItem()
                Generating one frequent patterns
            convert(value):
                To convert the given user specified value    
            getNeighbourItems(keySet):
                A function to get common neighbours of a itemSet
             mapNeighbours(file):
                A function to map items to their neighbours
    Executing the code on terminal :
    ---------------------------------
        Format:
        --------
            >>> python3 GPFPMiner.py <inputFile> <outputFile> <neighbourFile> <minSup> <maxPer>
        Examples:
        ---------
            >>> python3 GPFPMiner.py sampleTDB.txt output.txt sampleN.txt 0.5 0.3 (minSup & maxPer will be considered in percentage of database transactions)

           
    Sample run of importing the code :
    ------------------------------------
    .. code-block:: python
    
        import PAMI.geoReferencedPeridicFrequentPattern.GPFPMiner as alg

        obj = alg.GPFPMiner("sampleTDB.txt", "sampleN.txt", 5, 3)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of Geo Referenced Periodic-Frequent Patterns:", len(Patterns))

        obj.save("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
        The complete program was written by P.RaviKumar under the supervision of Professor Rage Uday Kiran.
    """

    _minSup = " "
    _maxPer = " "
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _nFile = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _sep = "\t"
    _lno = 0

    def __init__(self, iFile, nFile, minSup, maxPer, sep="\t"):
        super().__init__(iFile, nFile, minSup, maxPer, sep)
        self._NeighboursMap = {}

    def _creatingItemSets(self):
        """Storing the complete transactions of the database/input file in a database variable

        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            data, ts = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self._iFile['Transactions'].tolist()
            for i in range(len(data)):
                tr = [ts[i][0]]
                tr = tr + data[i]
                self._Database.append(tr)
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
                            line = line.rstrip()
                            temp = [i.strip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    # function to get frequent one pattern
    def _frequentOneItem(self):
        """Generating one frequent patterns"""

        candidate = {}
        for i in self._Database:
            self._lno += 1
            n = int(i[0])
            for j in i[1:]:
                if j not in candidate:
                    candidate[j] = [1, abs(0-n), n, [n]]
                else:
                    candidate[j][0] += 1
                    candidate[j][1] = max(candidate[j][1], abs(n - candidate[j][2]))
                    candidate[j][2] = n
                    candidate[j][3].append(n)
        self._minSup = self._convert(self._minSup)
        self._maxPer = self._convert(self._maxPer)
        #print(self._minSup, self._maxPer)
        self._tidList = {k: v[3] for k, v in candidate.items() if v[0] >= self._minSup and v[1] <= self._maxPer}
        candidate = {k: [v[0], v[1]] for k, v in candidate.items() if v[0] >= self._minSup and v[1] <= self._maxPer}
        plist = [key for key, value in sorted(candidate.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        return plist

    def _convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
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

    def _getSupportAndPeriod(self, timeStamps):
        """calculates the support and periodicity with list of timestamps

            :param timeStamps: timestamps of a pattern
            :type timeStamps: list
        """
        timeStamps.sort()
        cur = 0
        per = 0
        sup = 0
        for j in range(len(timeStamps)):
            per = max(per, timeStamps[j] - cur)
            if per > self._maxPer:
                return [0, 0]
            cur = timeStamps[j]
            sup += 1
        per = max(per, self._lno - cur)
        return [sup, per]

    def _save(self, prefix, suffix, tidSetX):
        """Saves the patterns that satisfy the periodic frequent property.

            :param prefix: the prefix of a pattern
            :type prefix: list or None
            :param suffix: the suffix of a patterns
            :type suffix: list
            :param tidSetX: the timestamp of a patterns
            :type tidSetX: list


        """
        if prefix == None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        val = self._getSupportAndPeriod(tidSetX)
        if val[0] >= self._minSup and val[1] <= self._maxPer:
            self._finalPatterns[tuple(prefix)] = val

    def _Generation(self, prefix, itemSets, tidSets):
        """Generates the patterns that satisfy the periodic frequent property.

            :param prefix: the prefix of a pattern
            :type prefix: list or None
            :param itemSets: the item sets of a patterns
            :type itemSets: list
            :param tidSets: the timestamp of a patterns
            :type tidSets: list


        """
        if len(itemSets) == 1:
            i = itemSets[0]
            tidI = tidSets[0]
            self._save(prefix, [i], tidI)
            return
        for i in range(len(itemSets)):
            itemX = itemSets[i]
            if itemX == None:
                continue
            tidSetX = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetX = [itemX]
            neighboursItemsI = self._getNeighbourItems(itemSets[i])
            for j in range(i + 1, len(itemSets)):
                neighboursItemsJ = self._getNeighbourItems(itemSets[i])
                if not itemSets[j] in neighboursItemsI:
                    continue
                itemJ = itemSets[j]
                tidSetJ = tidSets[j]
                y = list(set(tidSetX).intersection(tidSetJ))
                if len(y) >= self._minSup:
                    ne = list(set(neighboursItemsI).intersection(neighboursItemsJ))
                    x = []
                    x = x + [itemX]
                    x = x + [itemJ]
                    self._NeighboursMap[tuple(x)] = ne
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newPrefix = list(set(itemSetX)) + prefix
            self._Generation(newPrefix, classItemSets, classTidSets)
            self._save(prefix, list(set(itemSetX)), tidSetX)

    def _getNeighbourItems(self, keySet):
        """
            A function to get Neighbours of a item
            :param keySet:itemSet
            :type keySet:str or tuple
            :return: set of common neighbours 
            :rtype:set
        """
        itemNeighbours = self._NeighboursMap.keys()
        if isinstance(keySet, str):
            if self._NeighboursMap.get(keySet) is None:
                return []
            itemNeighbours = list(set(itemNeighbours).intersection(set(self._NeighboursMap.get(keySet))))
        if isinstance(keySet, tuple):
            keySet = list(keySet)
            for j in range(0, len(keySet)):
                i = keySet[j]
                itemNeighbours = list(set(itemNeighbours).intersection(set(self._NeighboursMap.get(i))))
        return itemNeighbours

    def mapNeighbours(self):
        """
            A function to map items to their Neighbours
        """
        self._NeighboursMap = {}
        if isinstance(self._nFile, _ab._pd.DataFrame):
            data = []
            if self._nFile.empty:
                print("its empty..")
            i = self._nFile.columns.values.tolist()
            if 'Neighbours' in i:
                data = self._nFile['Neighbours'].tolist()
            for i in data:
                self._NeighboursMap[i[0]] = i[1:]
        if isinstance(self._nFile, str):
            if _ab._validators.url(self._nFile):
                data = _ab._urlopen(self._nFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._NeighboursMap[temp[0]] = temp[1:]
            else:
                try:
                    with open(self._nFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._NeighboursMap[temp[0]] = temp[1:]
                except IOError:
                    print("File Not Found")
                    quit()

    def startMine(self):
        """Frequent pattern mining process will start from here"""

        # global items_sets, endTime, startTime
        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self.mapNeighbours()
        self._finalPatterns = {}
        plist = self._frequentOneItem()
        for i in range(len(plist)):
            itemX = plist[i]
            tidSetX = self._tidList[itemX]
            itemSetX = [itemX]
            itemSets = []
            tidSets = []
            neighboursItems = self._getNeighbourItems(plist[i])
            for j in range(i + 1, len(plist)):
                if not plist[j] in neighboursItems:
                    continue
                itemJ = plist[j]
                tidSetJ = self._tidList[itemJ]
                y1 = list(set(tidSetX).intersection(tidSetJ))
                if len(y1) >= self._minSup:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self._Generation(itemSetX, itemSets, tidSets)
            self._save(None, itemSetX, tidSetX)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Spatial Periodic Frequent patterns were generated successfully using SpatialEclat algorithm")

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
            pat = ""
            for i in a:
                pat += str(i) + "\t"
            data.append([pat, b[0], b[1]])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Period'])
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            pat = ""
            for i in x:
                pat += str(i) + "\t"
            patternsAndSupport = pat + ": " + str(y[0]) + ": " + str(y[1])
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns
    
    def printResults(self):
        """ This function is used to print the results
        """
        print("Total number of Spatial Periodic-Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 6 or len(_ab._sys.argv) == 7:
        if len(_ab._sys.argv) == 7:
            _ap = GPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])
        if len(_ab._sys.argv) == 6:
            _ap = GPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        _ap.startMine()
        print("Total number of Spatial Periodic-Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

