#  SpatialEclat is an Extension of ECLAT algorithm,which  stands for Equivalence Class Clustering and bottom-up
#  Lattice Traversal.It is one of the popular methods of Association Rule mining. It is a more efficient and
#  scalable version of the Apriori algorithm.
#
#  **Importing this algorithm into a python program**
#  ---------------------------------------------------
#
#     from PAMI.georeferencedFrequentPattern.basic import SpatialECLAT as alg
#
#     obj = alg.SpatialECLAT("sampleTDB.txt", "sampleN.txt", 5)
#
#     obj.startMine()
#
#     spatialFrequentPatterns = obj.getPatterns()
#
#     print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))
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

from PAMI.georeferencedFrequentPattern.basic import abstract as _ab


class SpatialECLAT(_ab._spatialFrequentPatterns):
    """
    Description:
    --------------
        Spatial Eclat is a Extension of ECLAT algorithm,which  stands for Equivalence Class Clustering and bottom-up
        Lattice Traversal.It is one of the popular methods of Association Rule mining. It is a more efficient and
        scalable version of the Apriori algorithm.

    Reference:
    -----------
        Rage, Uday & Fournier Viger, Philippe & Zettsu, Koji & Toyoda, Masashi & Kitsuregawa, Masaru. (2020).
        Discovering Frequent Spatial Patterns in Very Large Spatiotemporal Databases.

    Attributes :
    ---------------
        iFile : str
            Input file name or path of the input file
        nFile: str
            Name of Neighbourhood file name
        minSup: int or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
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
    ------------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
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
        dictKeysToInt(iList)
            Converting dictionary keys to integer elements
        eclatGeneration(cList)
            It will generate the combinations of frequent items
        generateSpatialFrequentPatterns(tidList)
            It will generate the combinations of frequent items from a list of items
        convert(value):
            To convert the given user specified value    
        getNeighbourItems(keySet):
            A function to get common neighbours of a itemSet
        mapNeighbours(file):
            A function to map items to their neighbours

    Executing the code on terminal :
    ----------------------------------
        Format:

            >>> python3 SpatialECLAT.py <inputFile> <outputFile> <neighbourFile> <minSup>
            
        Examples:

            >>> python3 SpatialECLAT.py sampleTDB.txt output.txt sampleN.txt 0.5 (minSup will be considered in percentage of database transactions)
            
    Sample run of importing the code :
    -----------------------------------
    .. code-block:: python

        from PAMI.georeferencedFrequentPattern.basic import SpatialECLAT as alg
        
        obj = alg.SpatialECLAT("sampleTDB.txt", "sampleN.txt", 5)

        obj.startMine()

        spatialFrequentPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))

        obj.save("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


    Credits:
    ---------
        The complete program was written by B.Sai Chitra under the supervision of Professor Rage Uday Kiran.
    """

    _minSup = float()
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

    def __init__(self, iFile, nFile, minSup, sep="\t"):
        super().__init__(iFile, nFile, minSup, sep)
        self._NeighboursMap = {}

    def _creatingItemSets(self):
        """Storing the complete transactions of the database/input file in a database variable


            """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
            if 'Patterns' in i:
                self._Database = self._iFile['Patterns'].tolist()
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

    # function to get frequent one pattern
    def _frequentOneItem(self):
        """Generating one frequent patterns"""
        self._finalPatterns = {}
        candidate = {}
        for i in range(len(self._Database)):
            for j in range(len(self._Database[i])):
                if self._Database[i][j] not in candidate:
                    candidate[self._Database[i][j]] = [i]
                else:
                    candidate[self._Database[i][j]] += [i]
        self._finalPatterns = {keys: value for keys, value in candidate.items() if len(value) >= self._minSup}

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

    @staticmethod
    def _dictKeysToInt(iList):
        """Converting dictionary keys to integer elements

        :param iList: Dictionary with patterns as keys and their support count as a value
        :type iList: dict
        :returns: list of integer patterns to represent dictionary keys
        :rtype: list
        """

        temp = []
        for ite in iList.keys():
            ite = [int(i) for i in ite.strip('[]').split('\t')]
            temp.append(ite)
            # print(sorted(temp))
        return sorted(temp)

    def _eclatGeneration(self, cList):
        """It will generate the combinations of frequent items

        :param cList :it represents the items with their respective transaction identifiers
        :type cList: dictionary
        :return: returning transaction dictionary
        :rtype: dict
        """
        # to generate all
        tidList = {}
        key = list(cList.keys())
        for i in range(0, len(key)):
            NeighboursItems = self._getNeighbourItems(key[i])
            for j in range(i + 1, len(key)):
                # print(c[key[i]],c[key[j]])
                if not key[j] in NeighboursItems:
                    continue
                intersectionList = list(set(cList[key[i]]).intersection(set(cList[key[j]])))
                itemList = []
                itemList += key[i]
                itemList += key[j]
                if len(intersectionList) >= self._minSup:
                    itemList.sort()
                    if tuple(itemList) not in tidList:
                        tidList[tuple(set(itemList))] = intersectionList
        return tidList

    def _generateSpatialFrequentPatterns(self, tidList):
        """It will generate the combinations of frequent items from a list of items

        :param tidList :it represents the items with their respective transaction identifiers
        :type tidList: dictionary
        :return: returning transaction dictionary
        :rtype: dict
        """
        tidList1 = {}
        if len(tidList) == 0:
            print("There are no more candidate sets")
        else:
            key = list(tidList.keys())
            for i in range(0, len(key)):
                NeighboursItems = self._getNeighbourItems(key[i])
                for j in range(i + 1, len(key)):
                    if not key[j] in NeighboursItems:
                        continue
                    intersectionList = list(set(tidList[key[i]]).intersection(set(tidList[key[j]])))
                    itemList = []
                    if len(intersectionList) >= self._minSup:
                        itemList += key[i], key[j]
                        itemList.sort()
                        tidList1[tuple(itemList)] = intersectionList

        return tidList1

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
            # print(keySet)
            for j in range(0, len(keySet)):
                i = keySet[j]
                itemNeighbours = list(set(itemNeighbours).intersection(set(self._NeighboursMap.get(i))))
        return itemNeighbours

    def _mapNeighbours(self):
        """
            A function to map items to their Neighbours
        """
        self._NeighboursMap = {}
        if isinstance(self._nFile, _ab._pd.DataFrame):
            data, items = [], []
            if self._nFile.empty:
                print("its empty..")
            i = self._nFile.columns.values.tolist()
            if 'item' in i:
                items = self._nFile['items'].tolist()
            if 'Neighbours' in i:
                data = self._nFile['Neighbours'].tolist()
            for k in range(len(items)):
                self._NeighboursMap[items[k]] = data[k]
            # print(self.Database)
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
        self._mapNeighbours()
        self._finalPatterns = {}
        self._frequentOneItem()
        frequentSet = self._generateSpatialFrequentPatterns(self._finalPatterns)
        for x, y in frequentSet.items():
            if x not in self._finalPatterns:
                self._finalPatterns[x] = y
        while 1:
            frequentSet = self._eclatGeneration(frequentSet)
            for x, y in frequentSet.items():
                if x not in self._finalPatterns:
                    self._finalPatterns[x] = y
            if len(frequentSet) == 0:
                break
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Spatial Frequent patterns were generated successfully using SpatialECLAT algorithm")

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
            pat = str()
            if type(a) == str:
                pat = a
            if type(a) == list:
                for i in a:
                    pat = pat + a + ' '
            data.append([pat.strip(), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            pat = str()
            if type(x) == str:
                pat = x 
            if type(x) == list:
                for i in x:
                    pat = pat + x + '\t'
            patternsAndSupport = pat.strip() + ":" + str(len(y))
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
        print("Total number of Spatial Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = SpatialECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = SpatialECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        print("Total number of Spatial Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
