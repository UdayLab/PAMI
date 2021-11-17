#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
from abstract import *


class UPItem:
    """
    A class to represent the UPItem

    Attribute :
    --------
    name: int
        name of item
    utility: int
        utility of item
    Methods :
    -------
    getUtility()
        method to get node utility
    setUtility()
        method to set node utility
    getName()
        method to get name of particular item
    """
    name = 0
    utility = 0

    def __init__(self, name, utility):
        self.name = name
        self.utility = utility

    def getUtility(self):
        """
            method to get node utility
        """
        return self.utility

    def setUtility(self, utility):
        """
            method to set node utility
             :param utility: the utility to set
             :type utility: int
        """
        self.utility = utility

    def getName(self):
        """
            method to get name for particular item
        """
        return self.name


class UPNode:
    """
        A class that represent UPNode
    Attribute :
    ---------
        itemId :int
            name of the item
        count: int
            represent the count of items
        nodeUtility: int
            Represent the utility of current node
        nodeLink: UPNode
            represent the link to next node with same itemid
        childs: list
            link to next node with same item Id (for the header table)
    Method:
    ------
        getChildWithId( name):
            method to get child node Return the immediate child of this node having a given name
    """
    itemId = -1
    count = 1
    nodeUtility = 0
    childs = []
    nodeLink = -1
    parent = -1

    def __init__(self):
        self.itemId = -1
        self.count = 1
        self.nodeUtility = 0
        self.childs = []
        self.nodeLink = -1
        self.parent = -1

    def getChildWithId(self, name):
        """
            method to get child node Return the immediate child of this node having a given name
            Parameters:
            ----------
            :param name: represent id of item
            :type name: int
        """
        for child in self.childs:
            if child.itemId == name:
                return child
        return -1


class UPTree:
    """
        A class to represent UPTree
    Attributes :
    -----------
        headerList: list
            List of items in the header table
        mapItemNodes: map
            Map that indicates the last node for each item using the node links
        root : UPNode
            root of the tree
        mapItemToLastNode: map
            List of pairs (item, Utility) of the header table
        hasMoreThanOnePath :bool
            Variable that indicate if the tree has more than one path

    Methods:
    --------
        addTransaction(transaction,rtu)
            To add a transaction (for initial construction)
        addLocalTransactio(localPath, pathUtility, mapItemToMinimumItemutility, pathCount)
            Add a transaction to the UP-Tree (for a local UP-Tree)
        insertNewNode(currentlocalNode, itemName, nodeUtility)
            Insert a new node in the UP-Tree as child of a parent node
        createHeaderList(mapItemToTwu)
            Method for creating the list of items in the header table, in descending order of TWU or path utility.
    """
    headerList = []
    hasMoreThanOnePath = False
    mapItemNodes = {}
    root = UPNode()
    mapItemToLastNode = {}

    def __init__(self):
        self.headerList = []
        self.hasMoreThanOnePath = False
        self.mapItemToLastNode = {}
        self.mapItemNodes = {}
        self.root = UPNode()

    def addTransaction(self, transaction, RTU):
        """
            A Method to add new Transaction to tree
            :param transaction: the reorganised transaction
            :type transaction: list
            :param RTU :reorganised transaction utility
            :type RTU: int
        """
        currentNode = self.root
        NumberOfNodes = 0
        RemainingUtility = 0
        # for idx, item in enumerate(transaction):
        #     itemName = item.name
        #     child = currentNode.getChildWithId(itemName)
        #     RemainingUtility += item.utility
        #     if child == -1:
        #         NumberOfNodes += 1
        #         nodeUtility = RemainingUtility
        #         currentNode = self.insertNewNode(currentNode, itemName, nodeUtility)
        #     else:
        #         child.count += 1
        #         child.nodeUtility += RemainingUtility
        #         currentNode = child
        for idx, item in enumerate(transaction):
            for k in range(idx + 1, len(transaction)):
                RemainingUtility += transaction[k].getUtility()
            itemName = item.name
            child = currentNode.getChildWithId(itemName)
            if child == -1:
                NumberOfNodes += 1
                nodeUtility = RTU - RemainingUtility
                RemainingUtility = 0
                currentNode = self.insertNewNode(currentNode, itemName, nodeUtility)
            else:
                currentNU = child.nodeUtility
                nodeUtility = currentNU + (RTU - RemainingUtility)
                RemainingUtility = 0
                child.count += 1
                child.nodeUtility = nodeUtility
                currentNode = child
        return NumberOfNodes

    def addLocalTransaction(self, localPath, pathUtility, mapItemToMinimumItemutility, pathCount):
        """
            A Method to add addLocalTransaction to tree

            :param localPath: The path to insert
            :type localPath: list
            :param pathUtility: the Utility of path
            :type pathUtility: int
            :param mapItemToMinimumItemutility: he map storing minimum item utility
            :type mapItemToMinimumItemutility: map
            :param pathCount: the Path count
            :type pathCount: int
        """
        currentLocalNode = self.root
        RemainingUtility = 0
        NumberOfNodes = 0
        # for item in localPath:
        #     RemainingUtility += mapItemToMinimumItemutility[item] * pathCount
        # for item in localPath:
        #     RemainingUtility -= mapItemToMinimumItemutility[item] * pathCount
        #     child = currentLocalNode.getChildWithId(item)
        #     if child == -1:
        #         NumberOfNodes += 1
        #         currentLocalNode = self.insertNewNode(currentLocalNode, item, pathUtility - RemainingUtility)
        #     else:
        #         child.count += 1
        #         child.nodeUtility += (pathUtility - RemainingUtility)
        #         currentLocalNode = child
        for idx, item in enumerate(localPath):
            for k in range(idx + 1, len(localPath)):
                search = localPath[k]
                RemainingUtility += mapItemToMinimumItemutility[search] * pathCount
            child = currentLocalNode.getChildWithId(item)
            if child == -1:
                NumberOfNodes += 1
                nodeUtility = pathUtility - RemainingUtility
                RemainingUtility = 0
                currentLocalNode = self.insertNewNode(currentLocalNode, item, nodeUtility)
            else:
                currentNU = child.nodeUtility
                nodeUtility = currentNU + (pathUtility - RemainingUtility)
                RemainingUtility = 0
                child.count += 1
                child.nodeUtility = nodeUtility
                currentLocalNode = child
        return NumberOfNodes

    def insertNewNode(self, currentlocalNode, itemName, nodeUtility):
        """
            A method to Insert a new node in the UP-Tree as child of a parent node
             :param currentlocalNode: The parent Node
             :type currentlocalNode: UPNode
             :param itemName: name of item in new Node
             :type itemName: int
             :param nodeUtility: Utility of new node
             :type nodeUtility: int
        """
        newNode = UPNode()
        newNode.itemId = itemName
        newNode.count = 1
        newNode.nodeUtility = nodeUtility
        newNode.parent = currentlocalNode
        currentlocalNode.childs.append(newNode)
        if not self.hasMoreThanOnePath and len(currentlocalNode.childs) > 1:
            self.hasMoreThanOnePath = True
        if itemName in self.mapItemNodes:
            lastNode = self.mapItemToLastNode[itemName]
            lastNode.nodeLink = newNode
            self.mapItemToLastNode[itemName] = newNode
        else:
            self.mapItemNodes[itemName] = newNode
            self.mapItemToLastNode[itemName] = newNode
        return newNode

    def createHeaderList(self, mapItemToTwu):
        """
            A Method for creating the list of items in the header table, in descending order of TWU or path utility.
            :param mapItemToTwu: the Utilities of each item
            :type mapItemToTwu: map
        """
        self.headerList = list(self.mapItemNodes.keys())
        self.headerList = sorted(self.headerList, key=lambda x: mapItemToTwu[x], reverse=True)


class UPGrowth(utilityPatterns):
    """
    UP-Growth is two-phase algorithm to mine High Utility Itemsets from transactional databases.

    Reference:
    ---------
        Vincent S. Tseng, Cheng-Wei Wu, Bai-En Shie, and Philip S. Yu. 2010. UP-Growth: an efficient algorithm for high utility itemset mining.
        In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '10).
        Association for Computing Machinery, New York, NY, USA, 253–262. DOI:https://doi.org/10.1145/1835804.1835839

    Attributes:
    ---------
        iFile : file
            Name of the input file to mine complete set of frequent patterns
        oFile : file
            Name of the output file to store complete set of frequent patterns
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        minUtil : int
            The user given minUtil
        NumberOfNodes : int
            Total number of nodes generated while building the tree
        ParentNumberOfNodes : int
           Total number of nodes required to build the parent tree
        MapItemToMinimumUtility : map
           A map to store the minimum utility of item in the database
        phuis : list
            A list to store the phuis
        MapItemToTwu : map
            A map to store the twu of each item in database

    Methods :
    -------
        startMine()
                Mining process will start from here
        getPatterns()
                Complete set of patterns will be retrieved with this function
        createLocalTree(tree, item)
            A Method to Construct conditional pattern base
        UPGrowth( tree, alpha)
            A Method to Mine UP Tree recursively
        PrintStats()
            A Method to print no.of phuis
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
    Executing the code on terminal :
    -------
        Format: python3 UPGrowth <inputFile> <outputFile> <Neighbours> <minUtil> <sep>
        Examples: python3 UPGrowth sampleTDB.txt output.txt sampleN.txt 35  (it will consider "\t" as separator)
                  python3 UPGrowth sampleTDB.txt output.txt sampleN.txt 35 , (it will consider "," as separator)
    Sample run of importing the code:
    -------------------------------

        from PAMI.highUtilityPatterns.basic import UPGrowth as alg
        obj=alg.UPGrowth("input.txt",35)
        obj.startMine()
        frequentPatterns = obj.getPatterns()
        print("Total number of Spatial Frequent Patterns:", len(frequentPatterns))
        obj.savePatterns("output")
        memUSS = obj.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = obj.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = obj.getRuntime()
        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
        The complete program was written by pradeep pallikila under the supervision of Professor Rage Uday Kiran.

    """

    maxMemory = 0
    startTime = 0
    endTime = 0
    minUtil = 0
    memoryUSS = float()
    memoryRSS = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    NumberOfNodes = 0
    ParentNumberOfNodes = 0
    MapItemToMinimumUtility = {}
    MapItemsetsToUtilities = defaultdict(int)
    phuis = []
    Database = []
    MapItemToTwu = {}
    sep = " "

    def __init__(self, iFile, minUtil, sep='\t'):
        super().__init__(iFile, minUtil, sep)

    def creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable
        """
        self.Database = []
        if isinstance(self.iFile, pd.DataFrame):
            timeStamp, data = [], []
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'Transactions' in i:
                data = self.iFile['Transactions'].tolist()
            if 'Utilities' in i:
                data = self.iFile['Patterns'].tolist()
            for i in range(len(data)):
                tr = [timeStamp[i]]
                tr.append(data[i])
                self.Database.append(tr)
            # print(self.Database)
        if isinstance(self.iFile, str):
            if validators.url(self.iFile):
                data = urlopen(self.iFile)
                for line in data:
                    line = line.decode("utf-8")
                    self.Database.append(line)
            else:
                try:
                    with open(self.iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            self.Database.append(line)
                except IOError:
                    print("File Not Found")
                    quit()

    def startMine(self):
        self.startTime = time.time()
        tree = UPTree()
        self.creatingItemSets()
        self.finalPatterns = {}
        for line in self.Database:
            line = line.split("\n")[0]
            transaction = line.strip().split(':')
            items = transaction[0].split(self.sep)
            transactionUtility = int(transaction[1])
            for item in items:
                Item = int(item)
                if Item in self.MapItemToTwu:
                    self.MapItemToTwu[Item] += transactionUtility
                else:
                    self.MapItemToTwu[Item] = transactionUtility
        for line in self.Database:
            line = line.split("\n")[0]
            transaction = line.strip().split(':')
            items = transaction[0].split(self.sep)
            utilities = transaction[2].split(self.sep)
            remainingUtility = 0
            revisedTransaction = []
            for idx, item in enumerate(items):
                Item = int(item)
                utility = int(utilities[idx])
                if self.MapItemToTwu[Item] >= self.minUtil:
                    element = UPItem(Item, utility)
                    revisedTransaction.append(element)
                    remainingUtility += utility
                    if Item in self.MapItemToMinimumUtility:
                        minItemUtil = self.MapItemToMinimumUtility[Item]
                        if minItemUtil >= utility:
                            self.MapItemToMinimumUtility[Item] = utility
                    else:
                        self.MapItemToMinimumUtility[Item] = utility
            revisedTransaction = sorted(revisedTransaction, key=lambda x: self.MapItemToTwu[x.name], reverse=True)
            self.ParentNumberOfNodes += tree.addTransaction(revisedTransaction, remainingUtility)
        tree.createHeaderList(self.MapItemToTwu)
        alpha = []
        self.finalPatterns = {}
        # print("number of nodes in parent tree", self.ParentNumberOfNodes)
        self.UPGrowth(tree, alpha)
        # self.phuis = sorted(self.phuis, key=lambda x: len(x))
        # print(self.phuis[0:10])
        for line in self.Database:
            line = line.split("\n")[0]
            transaction = line.strip().split(':')
            items = transaction[0].split(self.sep)
            utilities = transaction[2].split(self.sep)
            mapItemToUtility = {}
            for idx, item in enumerate(items):
                Item = int(item)
                utility = int(utilities[idx])
                if self.MapItemToTwu[Item] >= self.minUtil:
                    mapItemToUtility[Item] = utility
            for itemset in self.phuis:
                l = len(itemset)
                count = 0
                utility = 0
                for item in itemset:
                    item = int(item)
                    if item in mapItemToUtility:
                        utility += mapItemToUtility[item]
                        count += 1
                if count == l:
                    self.MapItemsetsToUtilities[tuple(itemset)] += utility

        for itemset in self.phuis:
            util = self.MapItemsetsToUtilities[tuple(itemset)]
            if util >= self.minUtil:
                s = ""
                for item in itemset:
                    s = s + str(item)
                    s = s + " "
                self.finalPatterns[s] = util
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = float()
        self.memoryRSS = float()
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("High Utility patterns were generated successfully using UPGrowth algorithm")

    def UPGrowth(self, tree, alpha):
        """
            A Method to Mine UP Tree recursively
            :param tree: UPTree to mine
            :type tree: UPTree
            :param alpha: prefix itemset
            :type alpha: list
        """
        for item in reversed(tree.headerList):
            localTree = self.createLocalTree(tree, item)
            node = tree.mapItemNodes[item]
            ItemTotalUtility = 0
            while node != -1:
                ItemTotalUtility += node.nodeUtility
                node = node.nodeLink
            if ItemTotalUtility >= self.minUtil:
                beta = alpha + [item]
                self.phuis.append(beta)
                # str1 = ' '.join(map(str, beta))
                # self.finalPatterns[str1] = ItemTotalUtility
                if len(localTree.headerList) > 0:
                    self.UPGrowth(localTree, beta)

    def createLocalTree(self, tree, item):
        """
            A Method to Construct conditional pattern base
            :param tree: the UPtree
            :type tree:UPTree
            :param item: item that need to construct conditional patterns
            :type item: int
        """
        prefixPaths = []
        path = tree.mapItemNodes[item]
        itemPathUtility = {}
        while path != -1:
            nodeUtility = path.nodeUtility
            if path.parent != -1:
                prefixPath = []
                prefixPath.append(path)
                ParentNode = path.parent
                while ParentNode.itemId != -1:
                    prefixPath.append(ParentNode)
                    itemName = ParentNode.itemId
                    if itemName in itemPathUtility:
                        itemPathUtility[itemName] += nodeUtility
                    else:
                        itemPathUtility[itemName] = nodeUtility
                    ParentNode = ParentNode.parent
                prefixPaths.append(prefixPath)
            path = path.nodeLink
        localTree = UPTree()
        for prefixPath in prefixPaths:
            pathUtility = prefixPath[0].nodeUtility
            pathCount = prefixPath[0].count
            localPath = []
            for i in range(1, len(prefixPath)):
                node = prefixPath[i]
                if itemPathUtility[node.itemId] >= self.minUtil:
                    localPath.append(node.itemId)
                else:
                    pathUtility -= pathCount * self.MapItemToMinimumUtility[node.itemId]
            localPath = sorted(localPath, key=lambda x: itemPathUtility[x], reverse=True)
            self.NumberOfNodes += localTree.addLocalTransaction(localPath, pathUtility, self.MapItemToMinimumUtility,
                                                                pathCount)
        localTree.createHeaderList(itemPathUtility)
        return localTree

    def PrintStats(self):
        """
            A Method to print no.of phuis
        """
        print('number of PHUIS are ' + str(len(self.phuis)))

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
            writer.write("%s\n" % patternsAndSupport)

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

        """
        Calculating the total amount of runtime taken by the mining process
            :return: returning total amount of runtime taken by the mining process
            :rtype: float
        """
        return self.endTime - self.startTime


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 4 or len(sys.argv) == 5:
        if len(sys.argv) == 5:
            ap = UPGrowth(sys.argv[1], int(sys.argv[3]), sys.argv[4])
        if len(sys.argv) == 4:
            ap = UPGrowth(sys.argv[1], int(sys.argv[3]))
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of huis:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        l = [500000]
        for i in l:
            ap = UPGrowth('/Users/Likhitha/Downloads/mushroom_utility_SPMF.txt', i, ' ')
            ap.startMine()
            Patterns = ap.getPatterns()
            print("Total number of huis:", len(Patterns))
            ap.savePatterns('/Users/Likhitha/Downloads/output')
            memUSS = ap.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = ap.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = ap.getRuntime()
            print("Total ExecutionTime in ms:", run)
        print("Error! The number of input parameters do not match the total number of parameters provided")