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
#      Copyright (C)  2021 Rage Uday Kiran

from PAMI.frequentPatternUsingOtherMeasure.abstract import *
import sys
import validators
from urllib.request import urlopen


class Node:
    """
        A class used to represent the node of frequentPatternTree

    Attributes:
    ----------
        itemId: int
            storing item of a node
        counter: int
            To maintain the support of node
        parent: node
            To maintain the parent of every node
        child: list
            To maintain the children of node
        nodeLink : node
            Points to the node with same itemId

    Methods:
    -------

        getChild(itemName)
            returns the node with same itemName from frequentPatternTree
    """

    def __init__(self):
        self.itemId = -1
        self.counter = 1
        self.parent = None
        self.child = []
        self.nodeLink = None

    def getChild(self, itemName):
        """ Retrieving the child from the tree

            :param itemName: name of the child
            :type itemName: list
            :return: returns the node with same itemName from frequentPatternTree
            :rtype: None or Node

        """
        for i in self.child:
            if i.itemId == itemName:
                return i
        return None


class Tree:
    """
        A class used to represent the frequentPatternGrowth tree structure

    Attributes:
    ----------
        headerList : list
            storing the list of items in tree sorted in ascending of their supports
        mapItemNodes : dictionary
            storing the nodes with same item name
        mapItemLastNodes : dictionary
            representing the map that indicates the last node for each item
        root : Node
            representing the root Node in a tree

    Methods:
    -------
        createHeaderList(items,minSup)
            takes items only which are greater than minSup and sort the items in ascending order
        addTransaction(transaction)
            creating transaction as a branch in frequentPatternTree
        fixNodeLinks(item,newNode)
            To create the link for nodes with same item
        printTree(Node)
            gives the details of node in frequentPatternGrowth tree
        addPrefixPath(prefix,port,minSup)
           It takes the items in prefix pattern whose support is >=minSup and construct a subtree
    """

    def __init__(self):
        self.headerList = []
        self.mapItemNodes = {}
        self.mapItemLastNodes = {}
        self.root = Node()

    def addTransaction(self, transaction):
        """adding transaction into tree

        :param transaction: it represents the one transactions in database
        :type transaction: list
        """

        # This method taken a transaction as input and returns the tree
        current = self.root
        for i in transaction:
            child = current.getChild(i)
            if not child:
                newNode = Node()
                newNode.itemId = i
                newNode.parent = current
                current.child.append(newNode)
                self.fixNodeLinks(i, newNode)
                current = newNode
            else:
                child.counter += 1
                current = child

    def fixNodeLinks(self, item, newNode):
        """Fixing node link for the newNode that inserted into frequentPatternTree

        :param item: it represents the item of newNode
        :type item: int
        :param newNode: it represents the newNode that inserted in frequentPatternTree
        :type newNode: Node

        """
        if item in self.mapItemLastNodes.keys():
            lastNode = self.mapItemLastNodes[item]
            lastNode.nodeLink = newNode
        self.mapItemLastNodes[item] = newNode
        if item not in self.mapItemNodes.keys():
            self.mapItemNodes[item] = newNode

    def printTree(self, root):
        """Print the details of Node in frequentPatternTree

        :param root: it represents the Node in frequentPatternTree
        :type root: Node

        """

        # this method is used print the details of tree
        if not root.child:
            return
        else:
            for i in root.child:
                print(i.itemId, i.counter, i.parent.itemId)
                self.printTree(i)

    def createHeaderList(self, mapSupport, minSup):
        """To create the headerList

        :param mapSupport: it represents the items with their supports
        :type mapSupport: dictionary
        :param minSup: it represents the minSup
        :param minSup: float
        """
        # the frequentPatternTree always maintains the header table to start the mining from leaf nodes
        t1 = []
        for x, y in mapSupport.items():
            if y >= minSup:
                t1.append(x)
        itemSetBuffer = [k for k, v in sorted(mapSupport.items(), key=lambda x: x[1], reverse=True)]
        self.headerList = [i for i in t1 if i in itemSetBuffer]

    def addPrefixPath(self, prefix, mapSupportBeta, minSup):
        """To construct the conditional tree with prefix paths of a node in frequentPatternTree

        :param prefix: it represents the prefix items of a Node
        :type prefix: list
        :param mapSupportBeta: it represents the items with their supports
        :param mapSupportBeta: dictionary
        :param minSup: to check the item meets with minSup
        :param minSup: float
        """
        # this method is used to add prefix paths in conditional trees of frequentPatternTree
        pathCount = prefix[0].counter
        current = self.root
        prefix.reverse()
        for i in range(0, len(prefix) - 1):
            pathItem = prefix[i]
            if mapSupportBeta.get(pathItem.itemId) >= minSup:
                child = current.getChild(pathItem.itemId)
                if not child:
                    newNode = Node()
                    newNode.itemId = pathItem.itemId
                    newNode.parent = current
                    newNode.counter = pathCount
                    current.child.append(newNode)
                    current = newNode
                    self.fixNodeLinks(pathItem.itemId, newNode)
                else:
                    child.counter += pathCount
                    current = child


class RSFPGrowth(frequentPatterns):
    """ 
        Algorithm to find all items with relative support from given dataset

    Reference:
    --------
        'Towards Efficient Discovery of Frequent Patterns with Relative Support' R. Uday Kiran and
               Masaru Kitsuregawa, http://comad.in/comad2012/pdf/kiran.pdf

    Attributes:
    ----------
        iFile : file
            Name of the Input file to mine complete set of frequent patterns
        oFile : file
            Name of the output file to store complete set of frequent patterns
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        minSup : float
            The user given minSup
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            it represents the total no of transactions
        tree : class
            it represents the Tree class
        itemSetCount : int
            it represents the total no of patterns
        finalPatterns : dict
            it represents to store the patterns
        itemSetBuffer : list
            it represents the store the items in mining
        maxPatternLength : int
           it represents the constraint for pattern length

    Methods:
    -------
        startMine()
            Mining process will start from here
        getFrequentPatterns()
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
        check(line)
            To check the delimiter used in the user input file
        creatingItemSets(fileName)
            Scans the dataset or dataframes and stores in list format
        frequentOneItem()
            Extracts the one-frequent patterns from transactions
        saveAllCombination(tempBuffer,s,position,prefix,prefixLength)
            Forms all the combinations between prefix and tempBuffer lists with support(s)
        saveItemSet(pattern,support)
            Stores all the frequent patterns with their respective support
        frequentPatternGrowthGenerate(frequentPatternTree,prefix,port)
            Mining the frequent patterns by forming conditional frequentPatternTrees to particular prefix item.
            mapSupport represents the 1-length items with their respective support
            
    Executing the code on terminal:
    -------
        Format:
        -------
            python3 RSFPGrowth.py <inputFile> <outputFile> <minSup> <minRatio>

        Examples:
        ---------
            python3 RSFPGrowth.py sampleDB.txt patterns.txt 0.23 0.2  (minSup will be considered in percentage of database transactions)

            python3 RSFPGrowth.py sampleDB.txt patterns.txt 3   0.2  (minSup will be considered in support count or frequency)

    
    Sample run of the importing code:
    -----------
        from PAMI.frequentPatternUsingOtherMeasures import RSFPGrowth as alg

        obj = alg.RSFPGrowth(iFile, minSup, minRatio)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    -------
        The complete program was written by Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.

        """

    startTime = float()
    endTime = float()
    minSup = str()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    mapSupport = {}
    lno = 0
    tree = Tree()
    itemSetBuffer = None
    fpNodeTempBuffer = []
    itemSetCount = 0
    maxPatternLength = 1000

    def __init__(self, iFile, minSup, minRatio, sep='\t'):
        super().__init__(iFile, minSup, minRatio, sep)
        self.finalPatterns = {}

    def creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


            """
        self.Database = []
        if isinstance(self.iFile, pd.DataFrame):
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'Transactions' in i:
                self.Database = self.iFile['Transactions'].tolist()
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

    def frequentOneItem(self):
        """Generating One frequent items sets

        """
        self.mapSupport = {}
        for i in self.Database:
            for j in i:
                if j not in self.mapSupport:
                    self.mapSupport[j] = 1
                else:
                    self.mapSupport[j] += 1

    def saveItemSet(self, prefix, prefixLength, support, ratio):
        """To save the frequent patterns mined form frequentPatternTree

        :param prefix: the frequent pattern
        :type prefix: list
        :param prefixLength: the length of a frequent pattern
        :type prefixLength: int
        :param support: the support of a pattern
        :type support:  int
        """

        sample = []
        for i in range(prefixLength):
            sample.append(prefix[i])
        self.itemSetCount += 1
        self.finalPatterns[tuple(sample)] = str(support)+" : "+str(ratio)

    def saveAllCombinations(self, tempBuffer, s, position, prefix, prefixLength):
        """Generating all the combinations for items in single branch in frequentPatternTree

        :param tempBuffer: items in a list
        :type tempBuffer: list
        :param s: support at leaf node of a branch
        :param position: the length of a tempBuffer
        :type position: int
        :param prefix: it represents the list of leaf node
        :type prefix: list
        :param prefixLength: the length of prefix
        :type prefixLength: int
        
        """
        max1 = 1 << position
        for i in range(1, max1):
            newPrefixLength = prefixLength
            for j in range(position):
                isSet = i & (1 << j)
                if isSet > 0:
                    prefix.insert(newPrefixLength, tempBuffer[j].itemId)
                    newPrefixLength += 1
            ratio = s/self.mapSupport[self.getMinItem(prefix, newPrefixLength)]
            if ratio >= self.minRatio:
                self.saveItemSet(prefix, newPrefixLength, s, ratio)

    def frequentPatternGrowthGenerate(self, frequentPatternTree, prefix, prefixLength, mapSupport, minConf):
        """Mining the fp tree

        :param frequentPatternTree: it represents the frequentPatternTree
        :type frequentPatternTree: class Tree
        :param prefix: it represents a empty list and store the patterns that are mined
        :type prefix: list
        :param param prefixLength: the length of prefix
        :type prefixLength: int
        :param mapSupport : it represents the support of item
        :type mapSupport : dictionary
        """
        singlePath = True
        position = 0
        s = 0
        if len(frequentPatternTree.root.child) > 1:
            singlePath = False
        else:
            currentNode = frequentPatternTree.root.child[0]
            while True:
                if len(currentNode.child) > 1:
                    singlePath = False
                    break
                self.fpNodeTempBuffer.insert(position, currentNode)
                s = currentNode.counter
                position += 1
                if len(currentNode.child) == 0:
                    break
                currentNode = currentNode.child[0]
        if singlePath is True:
            self.saveAllCombinations(self.fpNodeTempBuffer, s, position, prefix, prefixLength)
        else:
            for i in reversed(frequentPatternTree.headerList):
                item = i
                support = mapSupport[i]
                CminSup = max(self.minSup, support * self.minRatio)
                betaSupport = support              
                prefix.insert(prefixLength, item)
                max1 = self.getMinItem(prefix, prefixLength)
                if self.mapSupport[max1] > self.mapSupport[item]:
                    max1 = item
                ratio = support/self.mapSupport[max1]
                if ratio >= self.minRatio:
                    self.saveItemSet(prefix, prefixLength + 1, betaSupport, ratio)
                if prefixLength + 1 < self.maxPatternLength:
                    prefixPaths = []
                    path = frequentPatternTree.mapItemNodes.get(item)
                    mapSupportBeta = {}
                    while path is not None:
                        if path.parent.itemId != -1:
                            prefixPath = [path]
                            pathCount = path.counter
                            parent1 = path.parent
                            if mapSupport.get(parent1.itemId) >= CminSup:
                                while parent1.itemId != -1:
                                    mins = CminSup
                                    if mapSupport.get(parent1.itemId) >= mins:
                                        prefixPath.append(parent1)
                                        if mapSupportBeta.get(parent1.itemId) is None:
                                            mapSupportBeta[parent1.itemId] = pathCount
                                        else:
                                            mapSupportBeta[parent1.itemId] = mapSupportBeta[parent1.itemId] + pathCount
                                        parent1 = parent1.parent
                                    else:
                                        break
                                prefixPaths.append(prefixPath)
                        path = path.nodeLink
                    treeBeta = Tree()
                    for k in prefixPaths:
                        treeBeta.addPrefixPath(k, mapSupportBeta, self.minSup)
                    if len(treeBeta.root.child) > 0:
                        treeBeta.createHeaderList(mapSupportBeta, self.minSup)
                        self.frequentPatternGrowthGenerate(treeBeta, prefix, prefixLength + 1, mapSupportBeta, minConf)

    def convert(self, value):
        """
        to convert the type of user specified minSup value
        :param value: user specified minSup value
        :return: converted type
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

    def startMine(self):
        """
            main program to start the operation
        """

        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.minSup is None:
            raise Exception("Please enter the Minimum Support")
        self.creatingItemSets()
        self.minSup = self.convert(self.minSup)
        self.minRatio = float(self.minRatio)
        self.frequentOneItem()
        self.finalPatterns = {}
        self.mapSupport = {k: v for k, v in self.mapSupport.items() if v >= self.minSup}
        print(len(self.mapSupport), len(self.finalPatterns), len(self.Database))
        itemSetBuffer = [k for k, v in sorted(self.mapSupport.items(), key=lambda x: x[1], reverse=True)]
        for i in self.Database:
            transaction = []
            for j in i:
                if j in itemSetBuffer:
                    transaction.append(j)
            transaction.sort(key=lambda val: self.mapSupport[val], reverse=True)
            self.tree.addTransaction(transaction)
        self.tree.createHeaderList(self.mapSupport, self.minSup)
        if len(self.tree.headerList) > 0:
            self.itemSetBuffer = []
            self.frequentPatternGrowthGenerate(self.tree, self.itemSetBuffer, 0, self.mapSupport, self.minRatio)
        print("Relative support frequent patterns were generated successfully using RSFPGrowth algorithm")
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryRSS = float()
        self.memoryUSS = float()
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

    def getMinItem(self, prefix, prefixLength):
        """
            returns the minItem from prefix
        """
        minItem = prefix[0]
        for i in range(prefixLength):
            if self.mapSupport[minItem] > self.mapSupport[prefix[i]]:
                minItem = prefix[i]
        return minItem
    
    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process


        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self.endTime - self.startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataframe = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataframe

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            pattern = str()
            for i in x:
                pattern = pattern + i + " "
            s1 = str(pattern) + ": " + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        res = dict()
        for x, y in self.finalPatterns.items():
            pattern = str()
            for i in x:
                pattern = pattern + i + " "
            s1 = str(y)
            res[pattern] = s1

        return res


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = RSFPGrowth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = RSFPGrowth(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Frequent Patterns:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")        
