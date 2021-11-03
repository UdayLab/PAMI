import sys
from fpTree import Tree
from abstract import *

from collections import OrderedDict

class Node:
    """
    A class used to represent the node of frequentPatternTree

    Attributes
    ----------
        item : int
            Storing item of a node
        count : int
            To maintain the support of node
        children : dict
            To maintain the children of node
        prefix : list
            To maintain the prefix of node

    """
    def __init__(self,item,count,children):
        self.item = item
        self.count = count
        self.children = children
        self.prefix = []


class Tree:
    """
    A class used to represent the frequentPatternGrowth tree structure

    Attributes
    ----------
        root : Node
            The first node of the tree set to Null.
        nodeLink : dict
            Stores the nodes which shares same item

    Methods
    -------
        createTree(transaction,count)
            Adding transaction into the tree
        linkNode(node)
            Adding node to nodeLink
        createCPB(item,neighbour)
            Create conditional pattern base based on item and neighbour
        mergeTree(tree,fpList)
            Merge tree into yourself
        createTransactions(fpList)
            Create transactions from yourself
        getPattern(item,suffixItem,minSup,neighbour)
            Get frequent patterns based on suffixItem
        mining(minSup,isResponsible = lambda x:True,neighbpurhood=None)
            Mining yourself
    """

    def __init__(self):
        self.root = Node(None,0,{})
        self.nodeLink = OrderedDict()


    def createTree(self,transaction,count):
        """
        Create tree or add transaction into yourself.

        :param transaction: list
        :param count: int
        :return: Tree
        """
        current = self.root
        for item in transaction:
            if item not in current.children:
                current.children[item] = Node(item,count,{})
                current.children[item].prefix = transaction[0:transaction.index(item)]
                self.linkNode(current.children[item])
            else:
                current.children[item].count += count
            current = current.children[item]
        return self


    def linkNode(self, node):
        """
        Maintain link of node by adding node to nodeLink
        :param node: Node
        :return:
        """
        if node.item in self.nodeLink:
            self.nodeLink[node.item].append(node)
        else:
            self.nodeLink[node.item] = []
            self.nodeLink[node.item].append(node)


    def createCPB(self,item,neighbour):
        """
        Create conditional pattern base based on item and neighbour
        :param item: int
        :param neighbour: dict
        :return: Tree
        """
        pTree = Tree()
        for node in self.nodeLink[item]:
            node.prefix = [item for item in node.prefix if item in neighbour[node.item]]
            pTree.createTree(node.prefix,node.count)
        return pTree


    def mergeTree(self,tree,fpList):
        """
        Merge tree into yourself
        :param tree: Tree
        :param fpList: list
        :return: Tree
        """
        transactions = tree.createTransactions(fpList)
        for transaction in transactions:
            self.createTree(transaction,1)
        return self


    def createTransactions(self,fpList):
        """
        Create transactions that configure yourself
        :param fpList: list
        :return: list
        """
        transactions = []
        flist = [x for x in fpList if x in self.nodeLink]
        for item in reversed(flist):
            for node in self.nodeLink[item]:
                if node.count != 0:
                    transaction = node.prefix
                    transaction.append(node.item)
                    transactions.extend([transaction for i in range(node.count)])
                    current = self.root
                    for i in transaction:
                        current = current.children[i]
                        current.count -= node.count
        return transactions


    def getPattern(self,item,suffixItem,minSup,neighbour):
        """
        Get frequent patterns based on suffixItem
        :param item: int
        :param suffixItem: tuple
        :param minSup: int
        :param neighbour: dict
        :return: list
        """
        pTree = self.createCPB(item,neighbour)
        frequentItems = {}
        frequentPatterns = []
        for i in pTree.nodeLink.keys():
            frequentItems[i] = 0
            for node in pTree.nodeLink[i]:
                frequentItems[i] += node.count
        frequentItems = {key: value for key, value in frequentItems.items() if value >= minSup}
        for i in frequentItems:
            pattern = list(suffixItem)
            pattern.append(i)
            frequentPatterns.append((tuple(pattern),frequentItems[i]))
            frequentPatterns.extend(pTree.getPattern(i, tuple(pattern), minSup,neighbour))
        return frequentPatterns


    def mining(self,minSup,neighbpurhood=None):
        """
        Pattern mining on your own
        :param minSup: int
        :param isResponsible: function
        :param neighbpurhood: dict
        :return: list
        """
        frequentPatterns = []
        flist = sorted([item for item in self.nodeLink.keys()])
        for item in reversed(flist):
            frequentPatterns.extend(self.getPattern(item,(item,),minSup,neighbpurhood))
        return frequentPatterns


class FSPGrowth:
    """
    Attributes
    ----------
        iFile : file
            Input file name or path of the input file
        nFile : file
            Neighbourhood file name or path of the neighbourhood file
        oFile : file
            Name of the output file or the path of output file
        minSup : float
            The user can specify minSup either in count or proportion of database size.
        finalPatterns : dict
            Storing the complete set of patterns in a dictionary variable
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program

    Methods
    -------
        startMine()
            This function starts pattern mining.
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
        getNeighbour(string)
            This function changes string to tuple(neighbourhood).
        getFrequentItems(database)
            This function create frequent items from database.
        genCondTransaction(transaction, rank)
            This function generates conditional transaction for processing on each workers.
        getPartitionId(item)
            This function generates partition id
        mapNeighbourToNumber(neighbour, rank)
            This function maps neighbourhood to number.
            Because in this program, each item is mapped to number based on fpList so that it can be distributed.
            So the contents of neighbourhood must also be mapped to a number.
        getAllFrequentPatterns(data, fpList, ndata)
            This function generates all frequent patterns
    """

    minSup = float()
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    nFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    transaction = []
    neighbourList = {}

    def __init__(self, iFile, nFile, oFile, minSup):
        self.iFile = iFile
        self.minSup = int(minSup)
        self.nFile = nFile
        self.oFile = oFile

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

    def storePatternsInFile(self,oFile):
        """
        Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = oFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = x + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


    def readDatabase(self):
        """
        Read input file and neighborhood file
        In, addition, find frequent patterns that length is one. 
        """
        oneFrequentItem = {}
        with open(self.iFile, "r") as f:
            for line in f:
                l = line.rstrip().split('\t')
                l = [tuple(item.rstrip().split(' ')) for item in l]
                self.transaction.append(l)
                for item in l:
                    oneFrequentItem[item] = oneFrequentItem.get(item, 0) + 1
        oneFrequentItem = {key: value for key, value in oneFrequentItem.items() if value >= self.minSup}
        self.fpList = list(dict(sorted(oneFrequentItem.items(), key=lambda x: x[1], reverse=True)))
        self.finalPatterns = oneFrequentItem 

        with open(self.nFile,"r") as nf:
            for line in nf:
                l = line.rstrip().split('\t')
                key = tuple(l[0].rstrip().split(' '))
                for i in range(len(l)):
                    if i == 0:
                        self.neighbourList[key] = []
                    else:
                        self.neighbourList[key].append(tuple(l[i].rstrip().split(' ')))

    def sortTransaction(self):
        """
        Sort each transaction of self.transaction based on self.fpList
        """
        for i in range(len(self.transaction)):
            self.transaction[i] = [item for item in self.transaction[i] if item in self.fpList]
            self.transaction[i].sort(key=lambda value: self.fpList.index(value))
    
    def storePatternsInFile(self):
        """
        Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        s = ""
        s1 = ""
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            if type(x[0]) == str:
                s1 = "Point(" + x[0] + " " + x[1] + ")" + " : " + str(y) + "\n"
            else:
                for point in x:
                    s = "Point(" + str(point[0]) + " " + str(point[1]) + ")" + "\t"
                    s1 += s
                s1 += ": " + str(y) + "\n"
            writer.write(s1)
            s = ""
            s1 = ""
    
    

    def startMine(self):
        """
        start pattern mining from here
        """
        self.startTime = time.time()

        self.readDatabase()
        self.sortTransaction()
        FPTree = Tree()
        for trans in self.transaction:
            FPTree.createTree(trans,1)
        self.finalPatterns.update(dict(FPTree.mining(self.minSup,self.neighbourList)))
        self.storePatternsInFile()

        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("the number of frequent patterns : ",len(self.finalPatterns))
        print("runtime : ",self.getRuntime())
        print("memoryRSS : ",self.getMemoryRSS())
        print("memoryUSS : ",self.getMemoryUSS())


if __name__=="__main__":
    ap = FSPGrowth(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    ap.startMine()

