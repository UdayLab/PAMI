from PAMI.frequentSpatialPattern.basic.abstract import *
import sys
import pandas as pd


class SpatialEclat(spatialFrequentPatterns):
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
            minSup: float
                UserSpecified minimum support value. It has to be given in terms of count of total number of
                transactions in the input database/file
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
            storePatternsInFile(oFile)
                Complete set of frequent patterns will be loaded in to a output file
            getPatternsInDataFrame()
                Complete set of frequent patterns will be loaded in to a dataframe
            getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
            getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
            getRuntime()
                Total amount of runtime taken by the mining process will be retrieved from this function
            findDelimiter(line)
                Identifying the delimiter of the input file
            creatingItemSets(iFileName)
                Storing the complete transactions of the database/input file in a database variable
            frequentOneItem()
                Generating one frequent patterns
            dictKeysToInt(iList)
                Converting dictionary keys to integer elements
            eclatGeneration(cList)
                It will generate the combinations of frequent items
            generatespatialFrequentPatterns(tidList)
                It will generate the combinations of frequent items from a list of items
            convert(value):
                To convert the given user specified value    
            getNighboirItems(keyset):
                A function to get commom neighbours of a itemset
             mapNighbours(file):
                A function to map items to their neighbours

    Executing the code on terminal :
    ------------------------------
            Format: python3 SpatialEclat.py <inputFile> <outputFile> <neighbourFile> <minSup>
            Examples:
            python3 SpatialEclat.py sampleTDB.txt output.txt sampleN.txt 0.5 (minSup will be considered in percentage of database transactions)
            python3 SpatialEclat.py sampleTDB.txt output.txt sampleN.txt 3 (minSup will be considered in support count or frequency)
                                                                (it considers "\t" as separator)
                    SpatialEclat.py sampleTDB.txt output.txt sampleN.txt 3 , 
                                                                (it will consider "," as a separator)

    Sample run of importing the code :
    -------------------------------
        import SpatialEclat as alg
        obj = alg.SpatialEclat("sampleTDB.txt", "sampleN.txt", 5)

        obj.startMine()

        spatialFrequentPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))

        obj.storePatternsInFile("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


    Credits:
    -------
        The complete program was written by B.Sai Chitra under the supervision of Professor Rage Uday Kiran.
    """

    minSup = float()
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    nFile=" "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    sep="\t"

    def __init__(self, iFile, nFile, minSup, sep="\t"):
        super().__init__(iFile, nFile, minSup, sep)
        self.NighboursMap = {}

    def creatingItemSets(self, iFileName):
        """Storing the complete transactions of the database/input file in a database variable

            :param iFileName: user given input file/input file path
            :type iFileName: str
            """
        self.Database = []
        lineNumber = 0
        with open(iFileName, 'r', encoding='utf-8') as f:
            for line in f:
                # line.strip()
                if lineNumber == 0:
                    lineNumber += 1
                    # delimiter = self.findDelimiter([*line])
                    # li=[lineNumber]
                    li = line.split(self.sep)
                    li1 = [i.rstrip() for i in li]
                    self.Database.append([i.rstrip() for i in li1])
                    # else:
                    # self.Database.append(li)
                    # data.append([lineNumber,li1])
                else:
                    lineNumber += 1
                    li = line.split(self.sep)
                    # if delimiter==',':
                    li1 = [i.rstrip() for i in li]
                    self.Database.append(li1)

    # function to get frequent one pattern
    def frequentOneItem(self):
        """Generating one frequent patterns"""

        candidate = {}
        # global finalPatterns, minSup, Database
        # self.minSup = self.minSup
        for i in range(len(self.Database)):
            for j in range(len(self.Database[i])):
                if self.Database[i][j] not in candidate:
                    candidate[self.Database[i][j]] = [i]
                else:
                    candidate[self.Database[i][j]] += [i]
        self.finalPatterns = {keys: value for keys, value in candidate.items() if len(value) >= self.minSup}
        # print(candidate)

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

    @staticmethod
    def dictKeysToInt(iList):
        """Converting dictionary keys to integer elements

        :param iList: Dictionary with patterns as keys and their support count as a value
        :type iList: dict
        :returns: list of integer patterns to represent dictionary keys
        :rtype: list
        """

        temp = []
        for ite in iList.keys():
            ite = [int(i) for i in ite.strip('[]').split(',')]
            temp.append(ite)
            # print(sorted(temp))
        return sorted(temp)

    def eclatGeneration(self, cList):
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
            nighbousItems = self.getNighboirItems(key[i])
            for j in range(i + 1, len(key)):
                # print(c[key[i]],c[key[j]])
                if not key[j] in nighbousItems:
                    continue
                intersectionList = list(set(cList[key[i]]).intersection(set(cList[key[j]])))
                itemList = []
                itemList += key[i]
                itemList += key[j]
                if len(intersectionList) >= self.minSup:
                    itemList.sort()
                    if tuple(itemList) not in tidList:
                        tidList[tuple(set(itemList))] = intersectionList
        return tidList

    def generatespatialFrequentPatterns(self, tidList):
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
                nighbousItems = self.getNighboirItems(key[i])
                for j in range(i + 1, len(key)):
                    if not key[j] in nighbousItems:
                        continue
                    intersectionList = list(set(tidList[key[i]]).intersection(set(tidList[key[j]])))
                    itemList = []
                    if len(intersectionList) >= self.minSup:
                        itemList += key[i], key[j]
                        itemList.sort()
                        tidList1[tuple(itemList)] = intersectionList

        return tidList1

    def getNighboirItems(self, keyset):
        """
            A function to get Neighbours of a item
            :param keyset:itemset
            :type keyset:str or tuple
            :return: set of common neighbours 
            :rtype:set
        """
        itemNibours = self.NighboursMap.keys()
        if isinstance(keyset, str):
            if self.NighboursMap.get(keyset) is None:
                return []
            itemNibours = list(set(itemNibours).intersection(set(self.NighboursMap.get(keyset))))
        if isinstance(keyset, tuple):
            keyset = list(keyset)
            # print(keyset)
            for j in range(0, len(keyset)):
                i = keyset[j]
                itemNibours = list(set(itemNibours).intersection(set(self.NighboursMap.get(i))))
        return itemNibours

    def mapNighbours(self, name):
        """
            A function to map items to their Neighbours
            :param name: item name
            :type name: int
        """
        with open(name, 'r', encoding='utf-8') as f:
            for line in f:
                li = line.split(self.sep)
                item = li[0]
                nibs = li[1:]
                self.NighboursMap[item] = nibs

    def startMine(self):
        """Frequent pattern mining process will start from here"""

        # global items_sets, endTime, startTime
        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        iFileName = self.iFile
        self.creatingItemSets(iFileName)
        self.minSup = self.convert(self.minSup)
        self.mapNighbours(self.nFile)
        print(self.minSup)
        self.frequentOneItem()
        frequentSet = self.generatespatialFrequentPatterns(self.finalPatterns)
        # print("frequentSet",self.finalPatterns)
        for x, y in frequentSet.items():
            if x not in self.finalPatterns:
                self.finalPatterns[x] = y
        while 1:
            frequentSet = self.eclatGeneration(frequentSet)
            # print("frequentSet",frequentSet)
            for x, y in frequentSet.items():
                if x not in self.finalPatterns:
                    self.finalPatterns[x] = y
            if len(frequentSet) == 0:
                break
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Spatial Frequent patterns were generated successfully using SpatialEclat algorithm")

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

    def getPatternsInDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def storePatternsInFile(self, outFile):
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
            patternsAndSupport = pat + ": " + str(len(y))
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = SpatialEclat(sys.argv[1], sys.argv[3], sys.argv[4],sys.argv[5])
        if len(sys.argv) == 5:
            ap = SpatialEclat(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        spatialFrequentPatterns = ap.getPatterns()
        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
    else:
        print("Error!,Parameter doesn't match")