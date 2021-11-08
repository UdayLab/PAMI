import pandas as pd
import functools
from PAMI.highUtilitySpatialPattern.basic.abstract import *


class Element:
    """
    A class represents an Element of a utility list as used by the HDSHUI algorithm.

    Attributes :
    ----------
        ts : int
            keep tact of transaction id
        snu : int
            Spatial non closed itemSet utility
        remainingUtility : int
            Spatial non closed remaining utility
        pu : int
            prefix utility
        prevPos: int
            position of previous item in the list
    """

    def __init__(self, ts, snu, remainingUtility, pu, prevPos):
        self.ts = ts
        self.snu = snu
        self.remainingUtility = remainingUtility
        self.pu = pu
        self.prevPos = prevPos


class CUList:
    """
        A class represents a UtilityList as used by the HDSHUI algorithm.

    Attributes :
    ----------
        item: int
            item 
        sumSnu: long
            the sum of item utilities
        sumRemainingUtility: long
            the sum of remaining utilities
        sumCu : long
            the sum of closed utilities
        sumCru: long
            the sum of closed remaining utilities
        sumCpu: long
            the sum of closed prefix utilities
        elements: list
            the list of elements 

    Methods :
    -------
        addElement(element)
            Method to add an element to this utility list and update the sums at the same time.

    """

    def __init__(self, item):
        self.item = item
        self.sumSnu = 0
        self.sumRemainingUtility = 0
        self.sumCu = 0
        self.sumCru = 0
        self.sumCpu = 0
        self.elements = []

    def addElements(self, element):
        """
            A method to add new element to CUList
            :param element: element to be added to CUList
            :type element: Element
        """
        self.sumSnu += element.snu
        self.sumRemainingUtility += element.remainingUtility
        self.elements.append(element)


class Pair:
    """
        A class represent an item and its utility in a transaction
    """

    def __init__(self):
        self.item = 0
        self.utility = 0


class HDSHUIM(utilityPatterns):
    """
        Spatial High Utility ItemSet Mining (SHUIM) [3] is an important model in data
        mining with many real-world applications. It involves finding all spatially interesting itemSets having high value 
        in a quantitative spatio temporal database.

    Attributes :
    ----------
        iFile : str
            Name of the input file to mine complete set of frequent patterns
        oFile : str
            Name of the output file to store complete set of frequent patterns
        nFile: str
           Name of Neighbourhood items file
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        minUtil : int
            The user given minUtil
        mapFMAP: list
            EUCS map of the FHM algorithm
        candidates: int
            candidates generated
        huiCnt: int
            huis created
        neighbors: map
            keep track of neighbours of elements
        mapOfPMU: map
            a map to keep track of Probable Maximum utilty(PMU) of each item
    Methods :
    -------
            startMine()
                Mining process will start from here
            getPatterns()
                Complete set of patterns will be retrieved with this function
            savePatterns(oFile)
                Complete set of frequent patterns will be loaded in to a output file
            constructCUL(x, compactUList, st, minUtil, length, exneighbours)
                A method to construct CUL's database
            getPatternsAsDataFrame()
                Complete set of frequent patterns will be loaded in to a dataframe
            getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
            getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
            getRuntime()
                Total amount of runtime taken by the mining process will be retrieved from this function
            Explore_SearchTree(prefix, uList, exNeighbours, minUtil)
                A method to find all high utility itemSets
            updateClosed(x, compactUList, st, exCul, newT, ex, eyTs, length)
                A method to update closed values
            saveItemset(prefix, prefixLen, item, utility)
               A method to save itemSets
            updateElement(z, compactUList, st, exCul, newT, ex, duPrevPos, eyTs)
               A method to updates vales for duplicates


    Executing the code on terminal :
    -------
        Format:
            python3 HDSHUIM.py <inputFile> <outputFile> <Neighbours> <minUtil>

            python3 HDSHUIM.py <inputFile> <outputFile> <Neighbours> <minUtil> <separator>
        Examples:
            python3 HDSHUIM.py sampleTDB.txt output.txt sampleN.txt 35 (separator will be "\t" in both input and neighbourhood file)

            python3 HDSHUIM.py sampleTDB.txt output.txt sampleN.txt 35 , (separator will be "," in both input and neighbourhood file)

    Sample run of importing the code:
    -------------------------------
        
        from PAMI.highUtilityFrequentSpatialPattern.basic import HDSHUIM as alg

        obj=alg.HDSHUIM("input.txt","Neighbours.txt",35)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of Spatial High-Utility Patterns:", len(Patterns))

        obj.savePatterns("output")

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

    startTime = float()
    endTime = float()
    minSup = str()
    maxPer = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    nFile = " "
    minUtil = 0
    memoryUSS = float()
    memoryRSS = float()
    sep = "\t"

    def __init__(self, iFile1, neighb1, minUtil, sep="\t"):
        super().__init__(iFile1, neighb1, minUtil, sep)
        self.startTime = 0
        self.endTime = 0
        self.hui_cnt = 0
        self.candidates = 0
        self.mapOfPMU = {}
        self.mapFMAP = {}
        self.neighbors = {}
        self.finalPatterns = {}

    def compareItems(self, o1, o2):
        """
            A method to sort  list of huis in pmu ascending order
        """
        compare = self.mapOfPMU[o1.item] - self.mapOfPMU[o2.item]
        if compare == 0:
            return int(o1.item) - int(o2.item)
        else:
            return compare

    def startMine(self):
        """main program to start the operation
        """
        minUtil = self.minUtil
        self.startTime = time.time()
        with open(self.nFile, 'r') as file1:
            for line in file1:
                line = line.split("\n")[0]
                parts = line.split(self.sep)
                item = parts[0]
                neigh1 = set()
                for i in range(1, len(parts)):
                    neigh1.add(parts[i])
                self.neighbors[item] = neigh1
        with open(self.iFile, 'r') as file:
            for line in file:
                parts = line.split(":")
                items_str = (parts[0].split("\n")[0]).split(self.sep)
                utility_str = (parts[2].split("\n")[0]).split(self.sep)
                transUtility = int(parts[1])
                trans1 = set()
                for i in range(0, len(items_str)):
                    trans1.add(items_str[i])
                for i in range(0, len(items_str)):
                    item = items_str[i]
                    twu = self.mapOfPMU.get(item)
                    if (twu == None):
                        twu = int(utility_str[i])
                    else:
                        twu += int(utility_str[i])
                    self.mapOfPMU[item] = twu
                    if (self.neighbors.get(item) == None):
                        continue
                    neighbours2 = trans1.intersection(self.neighbors.get(item))
                    for item2 in neighbours2:
                        if (self.mapOfPMU.get(item2) == None):
                            self.mapOfPMU[item2] = int(utility_str[i])
                        else:
                            self.mapOfPMU[item2] += int(utility_str[i])

        listOfCUList = []
        hashTable = {}
        mapItemsToCUList = {}
        for item in self.mapOfPMU.keys():
            if (self.mapOfPMU.get(item) >= minUtil):
                uList = CUList(item)
                mapItemsToCUList[item] = uList
                listOfCUList.append(uList)
        listOfCUList.sort(key=functools.cmp_to_key(self.compareItems))
        ts = 1
        with open(self.iFile, 'r') as file:
            for line in file:
                parts = line.split(":")
                items = (parts[0].split("\n")[0]).split(self.sep)
                utilities = (parts[2].split("\n")[0]).split(self.sep)
                ru = 0
                newTwu = 0
                tx_key = []
                revisedTrans = []
                for i in range(0, len(items)):
                    pair = Pair()
                    pair.item = items[i]
                    pair.utility = int(utilities[i])
                    if self.mapOfPMU.get(pair.item) >= minUtil:
                        revisedTrans.append(pair)
                        tx_key.append(pair.item)
                        newTwu += pair.utility
                revisedTrans.sort(key=functools.cmp_to_key(self.compareItems))
                tx_key1 = tuple(tx_key)
                if len(revisedTrans) > 0:
                    if tx_key1 not in hashTable.keys():
                        hashTable[tx_key1] = len(mapItemsToCUList[revisedTrans[len(revisedTrans) - 1].item].elements)
                        for i in range(len(revisedTrans) - 1, -1, -1):
                            pair = revisedTrans[i]
                            cuListOfItems = mapItemsToCUList.get(pair.item)
                            element = Element(ts, pair.utility, ru, 0, 0)
                            if i > 0:
                                element.prevPos = len(mapItemsToCUList[revisedTrans[i - 1].item].elements)
                            else:
                                element.prevPos = -1
                            cuListOfItems.addElements(element)
                            ru += pair.utility
                    else:
                        pos = hashTable[tx_key1]
                        ru = 0
                        for i in range(len(revisedTrans) - 1, -1, -1):
                            cuListOfItems = mapItemsToCUList[revisedTrans[i].item]
                            cuListOfItems.elements[pos].snu += revisedTrans[i].utility
                            cuListOfItems.elements[pos].remainingUtility += ru
                            cuListOfItems.sumSnu += revisedTrans[i].utility
                            cuListOfItems.sumRemainingUtility += ru
                            ru += revisedTrans[i].utility
                            pos = cuListOfItems.elements[pos].prevPos
                # EUCS
                for i in range(len(revisedTrans) - 1, -1, -1):
                    pair = revisedTrans[i]
                    mapFMAPItem = self.mapFMAP.get(pair.item)
                    if mapFMAPItem is None:
                        mapFMAPItem = {}
                        self.mapFMAP[pair.item] = mapFMAPItem
                    for j in range(i + 1, len(revisedTrans)):
                        pairAfter = revisedTrans[j]
                        twuSUm = mapFMAPItem.get(pairAfter.item)
                        if twuSUm is None:
                            mapFMAPItem[pairAfter.item] = newTwu
                        else:
                            mapFMAPItem[pairAfter.item] = twuSUm + newTwu
                ts += 1
        exNeighbours = set(self.mapOfPMU.keys())
        # print(self.Neighbours)
        self.Explore_SearchTree([], listOfCUList, exNeighbours, minUtil)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

    def Explore_SearchTree(self, prefix, uList, exNeighbours, minUtil):
        """
            A method to find all high utility itemSets

            Attributes:
            -----------
            :parm prefix: it represent all items in prefix
            :type prefix :list
            :parm uList:projected Utility list
            :type uList: list
            :parm exNeighbours: keep track of common Neighbours
            :type exNeighbours: set
            :parm minUtil:user minUtil
            :type minUtil:int
        """
        for i in range(0, len(uList)):
            x = uList[i]
            if x.item not in exNeighbours:
                continue
            self.candidates += 1
            sortedPrefix = [0] * (len(prefix) + 1)
            sortedPrefix = prefix[0:len(prefix) + 1]
            sortedPrefix.append(x.item)
            if (x.sumSnu + x.sumCu >= minUtil) and (x.item in exNeighbours):
                self.saveItemset(prefix, len(prefix), x.item, x.sumSnu + x.sumCu)
            if x.sumSnu + x.sumCu + x.sumRemainingUtility + x.sumCru >= minUtil:  # U-Prune # and (x.item in exNeighbours)):
                ULIST = []
                for j in range(i, len(uList)):
                    if (uList[j].item in exNeighbours) and (self.neighbors.get(x.item) != None) and (
                            uList[j].item in self.neighbors.get(x.item)):
                        ULIST.append(uList[j])
                exULs = self.constructCUL(x, ULIST, -1, minUtil, len(sortedPrefix), exNeighbours)
                if self.neighbors.get(x.item) != None and exNeighbours != None:
                    set1 = exNeighbours.intersection(self.neighbors.get(x.item))
                    if exULs == None or set1 == None:
                        continue
                    self.Explore_SearchTree(sortedPrefix, exULs, set1, minUtil)

    def constructCUL(self, x, compactUList, st, minUtil, length, exNeighbours):
        """
            A method to construct CUL's database

            Attributes:
            -----------
            :parm x: Compact utility list
            :type x: Node
            :parm compactUList:list of Compact utility lists
            :type compactUList:list
            :parm st: starting pos of compactUList
            :type st:int
            :parm minUtil: user minUtil
            :type minUtil:int
            :parm length: length of x
            :type length:int
            :parm exNeighbours: common Neighbours
            :type exNeighbours: list
            :return: projected database of list X
            :rtype: list or set
        """
        exCul = []
        lau = []
        cUtil = []
        eyTs = []
        for i in range(0, len(compactUList)):
            uList = CUList(compactUList[i].item)
            exCul.append(uList)
            lau.append(0)
            cUtil.append(0)
            eyTs.append(0)
        sz = len(compactUList) - (st + 1)
        exSZ = sz
        for j in range(st + 1, len(compactUList)):
            mapOfTWUF = self.mapFMAP[x.item]
            if mapOfTWUF is not None:
                twuf = mapOfTWUF.get(compactUList[j].item)
                if twuf != None and twuf < minUtil or (not (exCul[j].item in exNeighbours)):
                    exCul[j] = None
                    exSZ = sz - 1
                else:
                    uList = CUList(compactUList[j].item)
                    exCul[j] = uList
                    eyTs[j] = 0
                    lau[j] = x.sumCu + x.sumCru + x.sumSnu + x.sumRemainingUtility
                    cUtil[j] = x.sumCu + x.sumCru
        hashTable = {}
        for ex in x.elements:
            newT = []
            for j in range(st + 1, len(compactUList)):
                if exCul[j] == None:
                    continue
                eyList = compactUList[j].elements
                while eyTs[j] < len(eyList) and eyList[eyTs[j]].ts < ex.ts:
                    eyTs[j] = eyTs[j] + 1
                if eyTs[j] < len(eyList) and eyList[eyTs[j]].ts == ex.ts:
                    newT.append(j)
                else:
                    lau[j] = lau[j] - ex.snu - ex.remainingUtility
                    if lau[j] < minUtil:
                        exCul[j] = None
                        exSZ = exSZ - 1
            if len(newT) == exSZ:
                self.updateClosed(x, compactUList, st, exCul, newT, ex, eyTs, length)
            else:
                if len(newT) == 0:
                    continue
                ru = 0
                newT1 = tuple(newT)
                if newT1 not in hashTable.keys():
                    hashTable[newT1] = len(exCul[newT[len(newT) - 1]].elements)
                    for i in range(len(newT) - 1, -1, -1):
                        cuListOfItems = exCul[newT[i]]
                        y = compactUList[newT[i]].elements[eyTs[newT[i]]]
                        element = Element(ex.ts, ex.snu + y.snu - ex.pu, ru, ex.snu, 0)
                        if i > 0:
                            element.prevPos = len(exCul[newT[i - 1]].elements)
                        else:
                            element.prevPos = -1
                        cuListOfItems.addElements(element)
                        ru += y.snu - ex.pu
                else:
                    dPrevPos = hashTable[newT1]
                    self.updateElement(x, compactUList, st, exCul, newT, ex, dPrevPos, eyTs)
            for j in range(st + 1, len(compactUList)):
                cUtil[j] = cUtil[j] + ex.snu + ex.remainingUtility
        filter_compactUList = []
        for j in range(st + 1, len(compactUList)):
            if cUtil[j] < minUtil or exCul[j] is None:
                continue
            else:
                if length > 1:
                    exCul[j].sumCu += compactUList[j].sumCu + x.sumCu - x.sumCpu
                    exCul[j].sumCru += compactUList[j].sumCru
                    exCul[j].sumCpu += x.sumCu
                filter_compactUList.append(exCul[j])
        return filter_compactUList

    def updateClosed(self, x, compactUList, st, exCul, newT, ex, eyTs, length):
        """
            A method to update closed values
            Attributes:
            -----------
            :parm x: Compact utility list
            :type x: list
            :parm compactUList:list of Compact utility lists
            :type compactUList:list
            :parm st: starting pos of compactUList
            :type st:int
            :parm newT:transaction to be updated
            :type newT:list
            :parm ex: element ex
            :type ex:element
            :parm eyTs:list of tss
            :type eyTs:ts
            :parm length: length of x
            :type length:int

        """
        remainingUtility = 0
        for j in range(len(newT) - 1, -1, -1):
            ey = compactUList[newT[j]]
            eyy = ey.elements[eyTs[newT[j]]]
            exCul[newT[j]].sumCu += ex.snu + eyy.snu - ex.pu
            exCul[newT[j]].sumCru += remainingUtility
            exCul[newT[j]].sumCpu += ex.snu
            remainingUtility = remainingUtility + eyy.snu - ex.pu

    def updateElement(self, z, compactUList, st, exCul, newT, ex, duPrevPos, eyTs):
        """
            A method to updates vales for duplicates

            Attributes:
            -----------
            :parm z: Compact utility list
            :type z: list
            :parm compactUList:list of Compact utility lists
            :type compactUList:list
            :parm st: starting pos of compactUList
            :type st:int
            :parm exCul:list of compactUList
            :type exCul:list
            :parm newT:transaction to be updated
            :type newT:list
            :parm ex: element ex
            :type ex:element
            :parm duPrevPos: position of z in exCul
            :type duPrevPos:int
            :parm eyTs:list of tss
            :type eyTs:ts
        """
        remainingUtility = 0
        pos = duPrevPos
        for j in range(len(newT) - 1, -1, -1):
            ey = compactUList[newT[j]]
            eyy = ey.elements[eyTs[newT[j]]]
            exCul[newT[j]].elements[pos].snu += ex.snu + eyy.snu - ex.pu
            exCul[newT[j]].sumSnu += ex.snu + eyy.snu - ex.pu
            exCul[newT[j]].elements[pos].remainingUtility += remainingUtility
            exCul[newT[j]].sumRemainingUtility += remainingUtility
            exCul[newT[j]].elements[pos].pu += ex.snu
            remainingUtility = remainingUtility + eyy.snu - ex.pu
            pos = exCul[newT[j]].elements[pos].prevPos

    def saveItemset(self, prefix, prefixLen, item, utility):
        """
         A method to save itemSets

         Attributes:
        -----------
        :parm prefix: it represent all items in prefix
        :type prefix :list
        :parm item:item
        :type item: int
        :parm utility:utility of itemset
        :type utility:int
        """
        self.hui_cnt += 1
        res = ""
        for i in range(0, prefixLen):
            res += str(prefix[i]) + " "
        res += str(item)
        res1 = str(utility)
        self.finalPatterns[res] = res1

    def getPatternsAsDataFrame(self):
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

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            patternsAndSupport = str(x) + " : " + str(y)
            writer.write("%s \n" % patternsAndSupport)

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


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:  # to  include a user specified separator
            ap = HDSHUIM(sys.argv[1], sys.argv[3], int(sys.argv[4]), sys.argv[5])
        if len(sys.argv) == 5:  # to consider "\t" as a separator
            ap = HDSHUIM(sys.argv[1], sys.argv[3], int(sys.argv[4]))
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Spatial High-Utility Patterns:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        ap = HDSHUIM('/home/apiiit-rkv/Downloads/ffsi_rainFallHighUtilityTransactionalDatabase.txt',
                     '/home/apiiit-rkv/Downloads/ffsi_neighborhoodRainFall_6.txt', 3000000, ' ')
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Spatial High-Utility Patterns:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
        print("Error! The number of input parameters do not match the total number of parameters provided")
