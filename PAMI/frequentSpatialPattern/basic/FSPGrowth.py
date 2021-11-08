from PAMI.frequentSpatialPattern.basic.abstract import *
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
    def __init__(self, item, count, children):
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
        mining(minSup,isResponsible = lambda x:True,neighbourhood=None)
            Mining yourself
    """

    def __init__(self):
        self.root = Node(None, 0, {})
        self.nodeLink = OrderedDict()

    def createTree(self, transaction, count):
        """
        Create tree or add transaction into yourself.

        :param transaction: list
        :param count: int
        :return: Tree
        """
        current = self.root
        for item in transaction:
            if item not in current.children:
                current.children[item] = Node(item, count, {})
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

    def createCPB(self, item, neighbour):
        """
        Create conditional pattern base based on item and neighbour
        :param item: int
        :param neighbour: dict
        :return: Tree
        """
        pTree = Tree()
        for node in self.nodeLink[item]:
            #print(node.item, neighbour[node.item])
            node.prefix = [item for item in node.prefix if item in neighbour.get(node.item)]
            pTree.createTree(node.prefix, node.count)
        return pTree

    def mergeTree(self, tree, fpList):
        """
        Merge tree into yourself
        :param tree: Tree
        :param fpList: list
        :return: Tree
        """
        transactions = tree.createTransactions(fpList)
        for transaction in transactions:
            self.createTree(transaction, 1)
        return self

    def createTransactions(self, fpList):
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

    def getPattern(self, item, suffixItem, minSup, neighbour):
        """
        Get frequent patterns based on suffixItem
        :param item: int
        :param suffixItem: tuple
        :param minSup: int
        :param neighbour: dict
        :return: list
        """
        pTree = self.createCPB(item, neighbour)
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
            frequentPatterns.append((tuple(pattern), frequentItems[i]))
            frequentPatterns.extend(pTree.getPattern(i, tuple(pattern), minSup, neighbour))
        return frequentPatterns

    def mining(self, minSup, neighbourhood=None):
        """
        Pattern mining on your own
        :param minSup: int
        :param neighbourhood: function
        :param neighbourhood: dict
        :return: list
        """
        frequentPatterns = []
        flist = sorted([item for item in self.nodeLink.keys()])
        for item in reversed(flist):
            frequentPatterns.extend(self.getPattern(item, (item,), minSup, neighbourhood))
        return frequentPatterns


class FSPGrowth(spatialFrequentPatterns):
    """
    Attributes:
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
        savePatterns(oFile)
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
    Executing the code on terminal :
    ------------------------------
        Format:
            python3 FSPGrowth.py <inputFile> <outputFile> <neighbourFile> <minSup>

        Examples:
            python3 FSPGrowth.py sampleTDB.txt output.txt sampleN.txt 0.5 (minSup will be considered in percentage of database transactions)

            python3 FSPGrowth.py sampleTDB.txt output.txt sampleN.txt 3 (minSup will be considered in support count or frequency)
                                                                (it considers "\t" as separator)

            python3 FSPGrowth.py sampleTDB.txt output.txt sampleN.txt 3 ','  (it will consider "," as a separator)

    Sample run of importing the code :
    -------------------------------

        from PAMI.frequentSpatialPattern.basic import FSPGrowth as alg

        obj = alg.FSPGrowth("sampleTDB.txt", "sampleN.txt", 5)

        obj.startMine()

        spatialFrequentPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))

        obj.savePatterns("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)


    Credits:
    -------
        The complete program was written by Yudai Masu under the supervision of Professor Rage Uday Kiran.
    """

    minSup = float()
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    nFile = " "
    oFile = " "
    sep = " "
    lno = 0
    memoryUSS = float()
    memoryRSS = float()
    transaction = []
    neighbourList = {}
    fpList = []

    '''def __init__(self, iFile, nFile, minSup, sep):
        self.iFile = iFile
        self.nFile = nFile
        self.minSup = minSup
        self.sep = sep'''

    def readDatabase(self):
        """
        Read input file and neighborhood file
        In, addition, find frequent patterns that length is one.
        """

        self.Database = []
        self.fpList = []
        self.finalPatterns = {}
        if isinstance(self.iFile, pd.DataFrame):
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'Transactions' in i:
                self.Database = self.iFile['Transactions'].tolist()
            self.lno = len(self.Database)
        if isinstance(self.iFile, str):
            if validators.url(self.iFile):
                data = urlopen(self.iFile)
                for line in data:
                    line.strip()
                    self.lno += 1
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.Database.append(temp)
            else:
                try:
                    with open(self.iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            self.lno += 1
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            self.Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()
        oneFrequentItem = {}
        for line in self.Database:
            for item in line:
                oneFrequentItem[item] = oneFrequentItem.get(item, 0) + 1
        oneFrequentItem = {key: value for key, value in oneFrequentItem.items() if value >= self.minSup}
        self.fpList = list(dict(sorted(oneFrequentItem.items(), key=lambda x: x[1], reverse=True)))
        print(len(self.fpList))
        self.finalPatterns = oneFrequentItem

        self.neighbourList = {}
        if isinstance(self.nFile, pd.DataFrame):
            data, items = [], []
            if self.nFile.empty:
                print("its empty..")
            i = self.nFile.columns.values.tolist()
            if 'item' in i:
                items = self.nFile['items'].tolist()
            if 'Neighbours' in i:
                data = self.nFile['Neighbours'].tolist()
            for k in range(len(items)):
                self.neighbourList[items[k][0]] = data[k]
            # print(self.Database)
        if isinstance(self.nFile, str):
            if validators.url(self.nFile):
                data = urlopen(self.nFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.neighbourList[temp[0]] = temp[1:]
            else:
                try:
                    with open(self.nFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            self.neighbourList[temp[0]] = temp[1:]
                except IOError:
                    print("File Not Found")
                    quit()
        '''with open(self.nFile, "r") as nf:
            for line in nf:
                l = line.rstrip().split('\t')
                key = tuple(l[0].rstrip().split(' '))
                for i in range(len(l)):
                    if i == 0:
                        self.neighbourList[key] = []
                    else:
                        self.neighbourList[key].append(tuple(l[i].rstrip().split(' ')))'''

    def sortTransaction(self):
        """
        Sort each transaction of self.Database based on self.fpList
        """
        for i in range(len(self.Database)):
            self.Database[i] = [item for item in self.Database[i] if item in self.fpList]
            self.Database[i].sort(key=lambda value: self.fpList.index(value))

    def convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self.lno * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self.lno * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
        start pattern mining from here
        """
        self.startTime = time.time()
        self.finalPatterns = {}
        self.readDatabase()
        self.minSup = self.convert(self.minSup)
        self.sortTransaction()
        FPTree = Tree()
        for trans in self.Database:
            FPTree.createTree(trans, 1)
        self.finalPatterns.update(dict(FPTree.mining(self.minSup, self.neighbourList)))
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = float()
        self.memoryRSS = float()
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Frequent Spatial Patterns successfully generated using FSPGrowth")

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

    def savePatterns(self, oFile):
        """
        Complete set of frequent patterns will be loaded in to a output file
        :param oFile: name of the output file
        :type oFile: file
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

    '''def savePatterns(self):
        """
        Complete set of frequent patterns will be loaded in to a output file
        
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
    '''


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = FSPGrowth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = FSPGrowth(sys.argv[1], sys.argv[3], sys.argv[4])
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
