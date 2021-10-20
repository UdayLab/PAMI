from PAMI.periodicFrequentSpatialPattern.abstract import *
import sys
import validators
from urllib.request import urlopen
import pandas as pd


class periodicSpatialEclat(spatialPeriodicFrequentPatterns):
    """ 
        Spatial Eclat is a Extension of ECLAT algorithm,which  stands for Equivalence Class Clustering and bottom-up
        Lattice Traversal.It is one of the popular methods of Association Rule mining. It is a more efficient and
        scalable version of the Apriori algorithm.
            ...
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
    -------
            startMine()
                Mining process will start from here
            getPatterns()
                Complete set of patterns will be retrieved with this function
            savePatterns(oFile)
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
    ------------------------------
        Format:
            python3 periodicSpatialEclat.py <inputFile> <outputFile> <neighbourFile> <minSup> <maxPer>
        Examples:
            python3 periodicSpatialEclat.py sampleTDB.txt output.txt sampleN.txt 0.5 0.3 (minSup & maxPer will be considered in percentage of database transactions)

            python3 periodicSpatialEclat.py sampleTDB.txt output.txt sampleN.txt 5 3 (minSup & maxPer will be considered in support count or frequency)
                                                                (it considers "\t" as separator)

            python3 periodicSpatialEclat.py sampleTDB.txt output.txt sampleN.txt 3 ',' (it will consider "," as a separator)

    Sample run of importing the code :
    -------------------------------

        import PAMI.periodicFrequentSpatialPattern.periodicSpatialEclat as alg

        obj = alg.periodicSpatialEclat("sampleTDB.txt", "sampleN.txt", 5, 3)

        obj.startMine()

        spatialPeriodicFrequentPatterns = obj.getPatterns()

        print("Total number of Periodic Spatial Frequent Patterns:", len(spatialPeriodicFrequentPatterns))

        obj.savePatterns("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
        The complete program was written by P. Likhitha under the supervision of Professor Rage Uday Kiran.
    """

    minSup = " "
    maxPer = " "
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    nFile = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    sep = "\t"
    lno = 0

    def __init__(self, iFile, nFile, minSup, maxPer, sep="\t"):
        super().__init__(iFile, nFile, minSup, maxPer, sep)
        self.NeighboursMap = {}

    def creatingItemSets(self, iFileName):
        """Storing the complete transactions of the database/input file in a database variable
            :param iFileName: user given input file/input file path
            :type iFileName: str
            """
        self.Database = []
        if isinstance(self.iFile, pd.DataFrame):
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'Transactions' in i:
                self.Database = self.iFile['Transactions'].tolist()
            if 'Patterns' in i:
                self.Database = self.iFile['Patterns'].tolist()
        if isinstance(self.iFile, str):
            if validators.url(self.iFile):
                data = urlopen(self.iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.Database.append(temp)
            else:
                try:
                    with open(self.iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            self.Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    # function to get frequent one pattern
    def frequentOneItem(self):
        """Generating one frequent patterns"""

        candidate = {}
        for i in self.Database:
            self.lno += 1
            for j in i:
                if j not in candidate:
                    candidate[j] = [1, abs(0-self.lno), self.lno, [self.lno]]
                else:
                    candidate[j][0] += 1
                    candidate[j][1] = max(candidate[j][1], abs(self.lno - candidate[j][2]))
                    candidate[j][2] = self.lno
                    candidate[j][3].append(self.lno)
        self.minSup = self.convert(self.minSup)
        self.maxPer = self.convert(self.maxPer)
        self.tidList = {k: v[3] for k, v in candidate.items() if v[0] >= self.minSup and v[1] <= self.maxPer}
        candidate = {k: [v[0], v[1]] for k, v in candidate.items() if v[0] >= self.minSup and v[1] <= self.maxPer}
        plist = [key for key, value in sorted(candidate.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        return plist

    def convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self.Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self.Database) * value)
            else:
                value = int(value)
        return value

    def getSupportAndPeriod(self, timeStamps):
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
            if per > self.maxPer:
                return [0, 0]
            cur = timeStamps[j]
            sup += 1
        per = max(per, self.lno - cur)
        return [sup, per]

    def save(self, prefix, suffix, tidSetX):
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
        val = self.getSupportAndPeriod(tidSetX)
        if val[0] >= self.minSup and val[1] <= self.maxPer:
            self.finalPatterns[tuple(prefix)] = val

    def Generation(self, prefix, itemSets, tidSets):
        if len(itemSets) == 1:
            i = itemSets[0]
            tidI = tidSets[0]
            self.save(prefix, [i], tidI)
            return
        for i in range(len(itemSets)):
            itemX = itemSets[i]
            if itemX == None:
                continue
            tidSetX = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetX = [itemX]
            neighboursItemsI = self.getNeighbourItems(itemSets[i])
            for j in range(i + 1, len(itemSets)):
                neighboursItemsJ = self.getNeighbourItems(itemSets[i])
                if not itemSets[j] in neighboursItemsI:
                    continue
                itemJ = itemSets[j]
                tidSetJ = tidSets[j]
                y = list(set(tidSetX).intersection(tidSetJ))
                if len(y) >= self.minSup:
                    ne = list(set(neighboursItemsI).intersection(neighboursItemsJ))
                    x = []
                    x = x + [itemX]
                    x = x + [itemJ]
                    self.NeighboursMap[tuple(x)] = ne
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newPrefix = list(set(itemSetX)) + prefix
            self.Generation(newPrefix, classItemSets, classTidSets)
            self.save(prefix, list(set(itemSetX)), tidSetX)

    def getNeighbourItems(self, keySet):
        """
            A function to get Neighbours of a item
            :param keySet:itemSet
            :type keySet:str or tuple
            :return: set of common neighbours 
            :rtype:set
        """
        itemNeighbours = self.NeighboursMap.keys()
        if isinstance(keySet, str):
            if self.NeighboursMap.get(keySet) is None:
                return []
            itemNeighbours = list(set(itemNeighbours).intersection(set(self.NeighboursMap.get(keySet))))
        if isinstance(keySet, tuple):
            keySet = list(keySet)
            for j in range(0, len(keySet)):
                i = keySet[j]
                itemNeighbours = list(set(itemNeighbours).intersection(set(self.NeighboursMap.get(i))))
        return itemNeighbours

    def mapNeighbours(self):
        """
            A function to map items to their Neighbours
        """
        self.NeighboursMap = []
        if isinstance(self.iFile, pd.DataFrame):
            data = []
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'Neighbours' in i:
                data = self.iFile['Neighbours'].tolist()
            for i in data:
                self.NeighboursMap[i[0]] = i[1:]
            # print(self.Database)
        if isinstance(self.iFile, str):
            if validators.url(self.iFile):
                data = urlopen(self.iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.NeighboursMap[temp[0]] = temp[1:]
            else:
                try:
                    with open(self.iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            self.NeighboursMap[temp[0]] = temp[1:]
                except IOError:
                    print("File Not Found")
                    quit()
    def startMine(self):
        """Frequent pattern mining process will start from here"""

        # global items_sets, endTime, startTime
        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        iFileName = self.iFile
        self.creatingItemSets(iFileName)
        self.minSup = self.convert(self.minSup)
        self.mapNeighbours()
        self.finalPatterns = {}
        plist = self.frequentOneItem()
        for i in range(len(plist)):
            itemX = plist[i]
            tidSetX = self.tidList[itemX]
            itemSetX = [itemX]
            itemSets = []
            tidSets = []
            neighboursItems = self.getNeighbourItems(plist[i])
            for j in range(i + 1, len(plist)):
                if not plist[j] in neighboursItems:
                    continue
                itemJ = plist[j]
                tidSetJ = self.tidList[itemJ]
                y1 = list(set(tidSetX).intersection(tidSetJ))
                if len(y1) >= self.minSup:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self.Generation(itemSetX, itemSets, tidSets)
            self.save(None, itemSetX, tidSetX)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = float()
        self.memoryRSS = float()
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Spatial Periodic Frequent patterns were generated successfully using SpatialEclat algorithm")

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function
        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self.memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function
        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self.memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process
        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self.endTime - self.startTime

    def getPatternsAsDataFrames(self):
        """Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support', 'Period'])
        return dataFrame

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            pat = ""
            for i in x:
                pat += str(i) + " "
            patternsAndSupport = pat + ": " + str(y[0]) + ": " + str(y[1])
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 6 or len(sys.argv) == 7:
        if len(sys.argv) == 7:
            ap = periodicSpatialEclat(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        if len(sys.argv) == 6:
            ap = periodicSpatialEclat(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        ap.startMine()
        spatialFrequentPatterns = ap.getPatterns()
        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
