import sys
from PAMI.localPeriodicPattern.basic.abstract import *

class Node:
    """
    A class used to represent the node of localPeriodicPatternTree
    ...
    Attributes
    ----------
    item : int
        storing item of a node
    parent : node
        To maintain the parent of every node
    child : list
        To maintain the children of node
    nodeLink : node
        To maintain the next node of node
    tidList : set
        To maintain timestamps of node

    Methods
    -------
    getChild(itemName)
    storing the children to their respective parent nodes
    """
    
    def __init__(self):
        self.item = -1
        self.parent = None
        self.child = []
        self.nodeLink = None
        self.tidList = set()

    def getChild(self, item):
        """
        :param item:
        :return: if node have node of item, then return it. if node don't have return []
        """
        for child in self.child:
            if child.item == item:
                return child
        return []


class Tree:
    """
        A class used to represent the frequentPatternGrowth tree structure

        ...
        Attributes
        ----------
        root : node
            Represents the root node of the tree
        nodeLinks : dictionary
            storing last node of each item
        firstNodeLink : dictionary
            storing first node of each item

        Methods
        -------
        addTransaction(transaction,timeStamp)
            creating transaction as a branch in frequentPatternTree
        fixNodeLinks(itemName, newNode)
            add newNode link after last node of item
        deleteNode(itemName)
            delete all node of item
        createPrefixTree(path,timeStampList)
            create prefix tree by path

    """
    def __init__(self):
        self.root = Node()
        self.nodeLinks = {}
        self.firstNodeLink = {}

    def addTransaction(self, transaction, tid):
        """
        add transaction into tree
        :param transaction: it represents the one transactions in database
        :type transaction: list
        :param tid: represents the timestamp of transaction
        :type tid: list
        """
        current = self.root
        for item in transaction:
            child = current.getChild(item)
            if not child:
                newNode = Node()
                newNode.item = item
                newNode.parent = current
                current.child.append(newNode)
                current = newNode
                self.fixNodeLinks(item, newNode)
            else:
                current = child
            current.tidList.add(tid)

    def fixNodeLinks(self, item, newNode):
        """
        fix node link
        :param item: it represents item name of newNode
        :type item: string
        :param newNode: it represents node which is added
        :type newNode: Node
        """
        if item in self.nodeLinks:
            lastNode = self.nodeLinks[item]
            lastNode.nodeLink = newNode
        self.nodeLinks[item] = newNode
        if item not in self.firstNodeLink:
            self.firstNodeLink[item] = newNode

    def deleteNode(self, item):
        """
        delete the node from tree
        :param item: it represents the item name of node
        :type item: str
        """
        deleteNode = self.firstNodeLink[item]
        parentNode = deleteNode.parent
        parentNode.child.remove(deleteNode)
        parentNode.child += deleteNode.child
        parentNode.tidList |= deleteNode.tidList
        for child in deleteNode.child:
            child.parent = parentNode
        while deleteNode.nodeLink:
            deleteNode = deleteNode.nodeLink
            parentNode = deleteNode.parent
            parentNode.child.remove(deleteNode)
            parentNode.child += deleteNode.child
            parentNode.tidList |= deleteNode.tidList
            for child in deleteNode.child:
                child.parent = parentNode

    def createPrefixTree(self, path, tidList):
        """
        create prefix tree by path
        :param path: it represents path to root from prefix node
        :type path: list
        :param tidList: it represents tid of each item
        :type tidList: list
        """
        currentNode = self.root
        for item in path:
            child = currentNode.getChild(item)
            if not child:
                newNode = Node()
                newNode.item = item
                newNode.parent = currentNode
                currentNode.child.append(newNode)
                currentNode = newNode
                self.fixNodeLinks(item, newNode)
            else:
                currentNode = child
            currentNode.tidList |= tidList



class LPPGrowth(localPeriodicPatterns):
    """

            Attributes:
            -----------
            iFile : str
                Input file name or path of the input file
            oFile : str
                Output file name or path of the output file
            maxPer : float
                User defined maxPer value.
            maxSoPer : float
                User defined maxSoPer value.
            minDur : float
                User defined minDur value.
            tsmin : int / date
                First time stamp of input data.
            tsmax : int / date
                Last time stamp of input data.
            startTime : float
                Time when start of execution the algorithm.
            endTime : float
                Time when end of execution the algorithm.
            finalPatterns : dict
                To store local periodic patterns and its PTL.
            tsList : dict
                To store items and its time stamp as bit vector.
            root : Tree
                It is root node of transaction tree of whole input data.
            PTL : dict
                Storing the item and its PTL.
            items : list
                Storing local periodic item list.
            :param sep: separator used to distinguish items from each other. The default separator is tab space.
            :type sep: str

            Methods
            -------
            findSeparator(line)
                Find the separator of the line which split strings.
            creteLPPlist()
                Create the local periodic patterns list from input data.
            createTSlist()
                Create the TSlist as bit vector from input data.
            generateLPP()
                Generate 1 length local periodic pattens by TSlist and execute depth first search.
            createLPPTree()
                Create LPPTree of local periodic item from input data.
            patternGrowth(tree, prefix, prefixPFList)
                Execute pattern growth algorithm. It is important function in this program.
            calculatePTL(tsList)
                Calculate PTL from input tsList as integer list.
            calculatePTLbit(tsList)
                Calculate PTL from input tsList as bit vector.
            startMine()
                Mining process will start from here.
            getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function.
            getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function.
            getRuntime()
                Total amount of runtime taken by the mining process will be retrieved from this function.
            getLocalPeriodicPatterns()
                return local periodic patterns and its PTL
            storePatternsInFile(oFile)
                Complete set of local periodic patterns will be loaded in to a output file.
            getPatternsInDataFrame()
                Complete set of local periodic patterns will be loaded in to a dataframe.

            Executing the code on terminal
            ------------------------------
            Format: python3 LPPMGrowth.py <inputFile> <outputFile> <maxPer> <minSoPer> <minDur>
            Examples: python3 LPPMGrowth.py sampleDB.txt patterns.txt 0.3 0.4 0.5
                      python3 LPPMGrowth.py sampleDB.txt patterns.txt 3 4 5

            Sample run of importing the code
            --------------------------------
            from PAMI.localPeriodicPattern.basic import LPPGrowth as alg
            obj = alg.LPPGrowth(iFile, maxPer, maxSoPer, minDur)
            obj.startMine()
            localPeriodicPatterns = obj.getLocalPeriodicPatterns()
            print(f'Total number of local periodic patterns: {len(localPeriodicPatterns)}')
            obj.storePatternsInFile(oFile)
            Df = obj.getPatternsInDataFrame()
            memUSS = obj.getMemoryUSS()
            print(f'Total memory in USS: {memUSS}')
            memRSS = obj.getMemoryRSS()
            print(f'Total memory in RSS: {memRSS}')
            runtime = obj.getRuntime()
            print(f'Total execution time in seconds: {runtime})

            Credits
            -------
            The complete program was written by So Nakamura under the supervision of Professor Rage Uday Kiran.
        """
    iFile = ' '
    oFile = ' '
    maxPer = float()
    maxSoPer = float()
    minDur = float()
    tsmin = 0
    tsmax = 0
    startTime = float()
    endTime = float()
    memoryUSS = float()
    memoryRSS = float()
    finalPatterns = {}
    tsList = {}
    root = Tree()
    PTL = {}
    items = []
    sep = ' '

    def findDelimiter(self, line):
        """Identifying the delimiter of the input file
            :param line: list of special characters may be used by a user to split the items in a input file
            :type line: list of string
            :returns: Delimited string used in the input file to split each item
            :rtype: string
            """
        l = [',', '*', '&', ' ', '%', '$', '#', '@', '!', '    ', '*', '(', ')']
        j = None
        for i in l:
            if i in line:
                return i
        return j

    def createLPPlist(self):
        """
        Create Local Periodic Pattern list from temporal data.
        """
        LPPList = {}
        PTL = {}
        start = {}
        tspre = {}
        with open(self.iFile, 'r') as f:
            line = f.readline()
            line = line.strip()
            separator = self.findDelimiter(line)
            # line = [item for item in line.split(separator)]
            line = [item for item in line.split(self.sep)]
            self.tsmin = int(line.pop(0))
            ts = self.tsmin
            for item in line:
                if item in LPPList:
                    per = ts - tspre[item]
                    if per <= self.maxPer and start == -1:
                        start = tspre[item]
                        soPer = self.maxSoPer
                    if start != -1:
                        soPer = max(0, soPer + per - self.maxPer)
                        if soPer > self.maxSoPer:
                            if tspre[item] - start[item] <= self.minDur:
                                PTL[item].add((start[item], tspre[item]))
                                LPPList[item] = PTL[item]
                            start[item] = -1
                else:
                    tspre[item] = ts
                    start[item] = -1
                    LPPList[item] = set()
            for line in f:
                line = line.strip()
                # line = [item for item in line.split(separator)]
                line = [item for item in line.split(self.sep)]
                ts = int(line.pop(0))
                for item in line:
                    if item in LPPList:
                        per = ts - tspre[item]
                        if per <= self.maxPer and start[item] == -1:
                            start[item] = tspre[item]
                            soPer = self.maxSoPer
                        if start[item] != -1:
                            soPer = max(0, soPer + per - self.maxPer)
                            if soPer > self.maxSoPer:
                                PTL[item].add((start[item], tspre[item]))
                                LPPList[item] = PTL[item]
                            start[item] = -1
                        tspre[item] = ts
                    else:
                        tspre[item] = ts
                        start[item] = -1
                        LPPList[item] = set()

    def createTSlist(self):
        """
        Create tsList as bit vector from temporal data.
        """
        with open(self.iFile, 'r') as f:
            count = 1
            bitVector = 0b1 << count
            bitVector = bitVector | 0b1
            line = f.readline()
            line = line.strip()
            separator = self.findDelimiter(line)
            # line = [item for item in line.split(separator)]
            line = [item for item in line.split(self.sep)]
            self.tsmin = int(line.pop(0))
            self.tsList = {item: bitVector for item in line}
            count += 1
            for line in f:
                bitVector = 0b1 << count
                bitVector = bitVector | 0b1
                line = line.strip()
                # line = [item for item in line.split(separator)]
                line = [item for item in line.split(self.sep)]
                ts = line.pop(0)
                for item in line:
                    if self.tsList.get(item):
                        different = abs(bitVector.bit_length() - self.tsList[item].bit_length())
                        self.tsList[item] = self.tsList[item] << different
                        self.tsList[item] = self.tsList[item] | 0b1
                    else:
                        self.tsList[item] = bitVector
                count += 1
            self.tsmax = int(ts)
            for item in self.tsList:
                different = abs(bitVector.bit_length() - self.tsList[item].bit_length())
                self.tsList[item] = self.tsList[item] << different
            self.maxPer = (count - 1) * self.maxPer
            self.maxSoPer = (count - 1) * self.maxSoPer
            self.minDur = (count - 1) * self.minDur

    def generateLPP(self):
        """
        Generate local periodic items from bit vector tsList.
        """
        I = set()
        PTL = {}
        for item in self.tsList:
            PTL[item] = set()
            ts = list(bin(self.tsList[item]))
            ts = ts[2:]
            start = -1
            currentTs = 1
            for t in ts[currentTs:]:
                if t == '0':
                    currentTs += 1
                    continue
                else:
                    tsPre = currentTs
                    currentTs += 1
                    break
            for t in ts[currentTs:]:
                if t == '0':
                    currentTs += 1
                    continue
                else:
                    per = currentTs - tsPre
                    if per <= self.maxPer and start == -1:
                        start = tsPre
                        soPer = self.maxSoPer
                    if start != -1:
                        soPer = max(0, soPer + per - self.maxPer)
                        if soPer > self.maxSoPer:
                            if tsPre - start >= self.minDur:
                                PTL[item].add((start, tsPre))
                            """else:
                                bitVector = 0b1 << currentTs
                                different = abs(self.tsList[item].bit_length() - bitVector.bit_length())
                                bitVector = bitVector | 0b1
                                bitVector = bitVector << different
                                self.tsList[item] = self.tsList[item] | bitVector"""
                            start = -1
                    tsPre = currentTs
                    currentTs += 1
            if start != -1:
                soPer = max(0, soPer + self.tsmax - tsPre - self.maxPer)
                if soPer > self.maxSoPer and tsPre - start >= self.minDur:
                    PTL[item].add((start, tsPre))
                """else:
                    bitVector = 0b1 << currentTs+1
                    different = abs(self.tsList[item].bit_length() - bitVector.bit_length())
                    bitVector = bitVector | 0b1
                    bitVector = bitVector << different
                    self.tsList[item] = self.tsList[item] | bitVector"""
                if soPer <= self.maxSoPer and self.tsmax - start >= self.minDur:
                    PTL[item].add((start, self.tsmax))
                """else:
                    bitVector = 0b1 << currentTs+1
                    different = abs(self.tsList[item].bit_length() - bitVector.bit_length())
                    bitVector = bitVector | 0b1
                    bitVector = bitVector << different
                    self.tsList[item] = self.tsList[item] | bitVector"""
        self.PTL = {k: v for k,v in PTL.items() if len(v) > 0}
        self.items = list(self.PTL.keys())

    def createLPPTree(self):
        """
        Create transaction tree of local periodic item from input data.
        """
        with open(self.iFile, 'r') as f:
            line = f.readline()
            line = line.strip()
            separator = self.findDelimiter(line)
            line = line.split(separator)
            ts = int(line[0])
            tempTransaction = [item for item in line if item in self.items]
            transaction = sorted(tempTransaction, key=lambda x: len(self.PTL[x]), reverse=True)
            self.root.addTransaction(transaction, ts)
            for line in f:
                line = line.strip()
                transaction = line.split(separator)
                tid = int(transaction.pop(0))
                tempTransaction = [item for item in transaction if item in self.items]
                transaction = sorted(tempTransaction, key=lambda x: len(self.PTL[x]), reverse=True)
                self.root.addTransaction(transaction, tid)

    def patternGrowth(self, tree, prefix, prefixPFList):
        """
        Create prefix tree and prefixPFList. Store finalPatterns and its PTL.
        :param tree: The root node of prefix tree.
        :type tree: Node
        :param prefix: Prefix item list.
        :type prefix: list
        :param prefixPFList: tsList of prefix patterns.
        :type prefixPFList: dict
        """
        items = list(prefixPFList)
        if not prefix:
            items = reversed(items)
        for item in items:
            prefixCopy = prefix.copy()
            prefixCopy.append(item)
            PFList = {}
            prefixTree = Tree()
            prefixNode = tree.firstNodeLink[item]
            tidList = prefixNode.tidList
            path = []
            currentNode = prefixNode.parent
            while currentNode.item != -1:
                path.insert(0, currentNode.item)
                currentNodeItem = currentNode.item
                if currentNodeItem in PFList:
                    PFList[currentNodeItem] |= tidList
                else:
                    PFList[currentNodeItem] = tidList
                currentNode = currentNode.parent
            prefixTree.createPrefixTree(path, tidList)
            while prefixNode.nodeLink:
                prefixNode = prefixNode.nodeLink
                tidList = prefixNode.tidList
                path = []
                currentNode = prefixNode.parent
                while currentNode.item != -1:
                    path.insert(0, currentNode.item)
                    currentNodeItem = currentNode.item
                    if currentNodeItem in PFList:
                        PFList[currentNodeItem] = PFList[currentNodeItem] | tidList
                    else:
                        PFList[currentNodeItem] = tidList
                    currentNode = currentNode.parent
                prefixTree.createPrefixTree(path, tidList)
            if len(prefixCopy) == 1:
                self.finalPatterns[prefixCopy[0]] = self.calculatePTLbit(self.tsList[item])
            else:
                self.finalPatterns[tuple(prefixCopy)] = self.calculatePTL(prefixPFList[item])
            candidateItems = list(PFList)
            for i in candidateItems:
                PTL = self.calculatePTL(PFList[i])
                if len(PTL) == 0:
                    prefixTree.deleteNode(i)
                    del PFList[i]
            if PFList:
                self.patternGrowth(prefixTree, prefixCopy, PFList)


    def calculatePTL(self, tslist):
        """
        Calculate PTL from input tsList as integer list/
        :param tslist: It is tslist which store time stamp as integer.
        :type tslist: list
        :return: PTL
        """
        start = -1
        PTL = set()
        tslist = sorted(tslist)
        tspre = tslist[0]
        for ts in tslist[1:]:
            per = ts - tspre
            if per <= self.maxPer and start == -1:
                start = tspre
                soPer = self.maxSoPer
            if start != -1:
                soPer = max(0, soPer + per - self.maxPer)
                if soPer > self.maxSoPer:
                    if tspre - start >= self.minDur:
                        PTL.add((start, tspre))
                    start = -1
            tspre = ts
        if start != -1:
            soPer = max(0, soPer + self.tsmax - tspre - self.maxPer)
            if soPer > self.maxSoPer and tspre - start >= self.minDur:
                PTL.add((start, tspre))
            if soPer <= self.maxSoPer and self.tsmax - start >= self.minDur:
                PTL.add((start, self.tsmax))
        return PTL

    def calculatePTLbit(self, tsList):
        """
        Calculate PTL from input tsList as bit vector.
        :param tsList: It is tsList which store time stamp as bit vector.
        :type tsList: list
        :return: PTL
        """
        tsList = list(bin(tsList))
        tsList = tsList[2:]
        start = -1
        currentTs = 1
        PTL = set()
        for ts in tsList[currentTs:]:
            if ts == '0':
                currentTs += 1
                continue
            else:
                tsPre = currentTs
                currentTs += 1
                break
        for ts in tsList[currentTs:]:
            if ts == '0':
                currentTs += 1
                continue
            else:
                per = currentTs - tsPre
                if per <= self.maxPer and start == -1:
                    start = tsPre
                    soPer = self.maxSoPer
                if start != -1:
                    soPer = max(0, soPer + per - self.maxPer)
                    if soPer > self.maxSoPer:
                        if tsPre - start >= self.minDur:
                            PTL.add((start, tsPre))
                        start = -1
                tsPre = currentTs
                currentTs += 1
        if start != -1:
            soPer = max(0, soPer + self.tsmax - tsPre - self.maxPer)
            if soPer > self.maxSoPer and tsPre - start >= self.minDur:
                PTL.add((start, tsPre))
            if soPer <= self.maxSoPer and self.tsmax - start >= self.minDur:
                PTL.add((start, tsPre))
        return PTL

    def startMine(self):
        """
        Mining process start from here.
        """
        self.startTime = time.time()
        self.createTSlist()
        self.generateLPP()
        self.createLPPTree()
        self.patternGrowth(self.root, [], self.items)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

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
        """Storing final local periodic patterns in a dataframe

        :return: returning local periodic patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'PTL'])
        return dataFrame

    def storePatternsInFile(self, outFile):
        """Complete set of local periodic patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            writer.write(f'{x} : {y}\n')
            # patternsAndPTL = x + ":" + y
            # writer.write("%s \n" % patternsAndPTL)

    def getLocalPeriodicPatterns(self):
        """ Function to send the set of local periodic patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == '__main__':
    if len(sys.argv) == 6:
        ap = LPPGrowth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        ap.startMine()
        localPeriodicPatterns = ap.getLocalPeriodicPatterns()
        print(f"Total number of Frequent Patterns: {len(localPeriodicPatterns)}")
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print(f'Total Memory in USS: {memUSS}')
        memRSS = ap.getMemoryRSS()
        print(f'Total Memory in RSS: {memRSS}')
        run = ap.getRuntime()
        print(f'Total ExecutionTime in seconds: {run}')
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

