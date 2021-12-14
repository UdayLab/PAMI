#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation,  either version 3 of the License,  or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not,  see <https://www.gnu.org/licenses/>.

from PAMI.uncertainFrequentPattern.basic import abstract as _fp

_minSup = float()
_fp._sys.setrecursionlimit(20000)
_finalPatterns = {}


class _Item:
    """
    A class used to represent the item with probability in transaction of dataset
    ...
        Attributes:
        __________
            item : int or word
                Represents the name of the item
            probability : float
                Represent the existential probability(likelihood presence) of an item
    """

    def __init__(self, item, probability):
        self.item = item
        self.probability = probability


class _Node(object):
    """
        A class used to represent the node of frequentPatternTree
        ...
        Attributes:
        ----------
            item : int
                storing item of a node
            probability : int
                To maintain the expected support of node
            parent : node
                To maintain the parent of every node
            children : list
                To maintain the children of node
        Methods:
        -------
            addChild(itemName)
                storing the children to their respective parent nodes
    """

    def __init__(self, item, children):
        self.item = item
        self.probability = 1
        self.maxPrefixProbability = 1
        self.p = 1
        self.children = children
        self.parent = None

    def addChild(self, node):
        self.children[node.item] = node
        node.parent = self


def printTree(root):
    for x, y in root.children.items():
        print(x, y.item, y.probability, y.parent.item, y.tids, y.maxPrefixProbability)
        printTree(y)


class _Tree(object):
    """
    A class used to represent the frequentPatternGrowth tree structure
    ...
    Attributes:
    ----------
        root : Node
            Represents the root node of the tree
        summaries : dictionary
            storing the nodes with same item name
        info : dictionary
            stores the support of items
    Methods:
    -------
        addTransaction(transaction)
            creating transaction as a branch in frequentPatternTree
        addConditionalTransaction(prefixPaths, supportOfItems)
            construct the conditional tree for prefix paths
        conditionalPatterns(Node)
            generates the conditional patterns from tree for specific node
        conditionalTransactions(prefixPaths,Support)
                takes the prefixPath of a node and support at child of the path and extract the frequent items from
                prefixPaths and generates prefixPaths with items which are frequent
        remove(Node)
                removes the node from tree once after generating all the patterns respective to the node
        generatePatterns(Node)
            starts from the root node of the tree and mines the frequent patterns
    """

    def __init__(self):
        self.root = _Node(None, {})
        self.summaries = {}
        self.info = {}

    def addTransaction(self, transaction):
        """adding transaction into tree
                :param transaction : it represents the one transactions in database
                :type transaction : list
                        """
        currentNode = self.root
        k = 0
        for i in range(len(transaction)):
            k += 1
            if transaction[i].item not in currentNode.children:
                newNode = _Node(transaction[i].item, {})
                newNode.k = k
                newNode.prefixProbability = transaction[i].probability
                l1 = i - 1
                temp = []
                while l1 >= 0:
                    temp.append(transaction[l1].probability)
                    l1 -= 1
                if len(temp) == 0:
                    newNode.probability = round(transaction[i].probability, 2)
                else:
                    newNode.probability = round(max(temp) * transaction[i].probability, 2)
                currentNode.addChild(newNode)
                if transaction[i].item in self.summaries:
                    self.summaries[transaction[i].item].append(newNode)
                else:
                    self.summaries[transaction[i].item] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i].item]
                currentNode.prefixProbability = max(transaction[i].probability, currentNode.prefixProbability)
                currentNode.k = k
                l1 = i - 1
                temp = []
                while l1 >= 0:
                    temp.append(transaction[l1].probability)
                    l1 -= 1
                if len(temp) == 0:
                    currentNode.probability += round(transaction[i].probability, 2)
                else:
                    nn = max(temp) * transaction[i].probability
                    currentNode.probability += round(nn, 2)

    def addConditionalTransaction(self, transaction, sup, second):
        """constructing conditional tree from prefixPaths
            :param transaction: it represents the one transactions in database
            :type transaction: list
            :param sup: support of prefixPath taken at last child of the path
            :type sup: int
            :param second: the second probability of the node
            :type second: float
        """
        currentNode = self.root
        k = 0
        for i in range(len(transaction)):
            k += 1
            if transaction[i] not in currentNode.children:
                newNode = _Node(transaction[i], {})
                newNode.k = k
                newNode.prefixProbability = second
                newNode.probability = sup
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
                currentNode.k = k
                currentNode.prefixProbability = max(currentNode.prefixProbability, second)
                currentNode.probability += sup

    def conditionalPatterns(self, alpha):
        """generates all the conditional patterns of respective node
            :param alpha : it represents the Node in tree
            :type alpha : _Node
        """
        finalPatterns = []
        sup = []
        second = []
        for i in self.summaries[alpha]:
            s = i.probability
            s1 = i.maxPrefixProbability
            set2 = []
            while i.parent.item is not None:
                set2.append(i.parent.item)
                i = i.parent
            if len(set2) > 0:
                set2.reverse()
                finalPatterns.append(set2)
                second.append(s1)
                sup.append(s)
        finalPatterns, support, info = self.conditionalTransactions(finalPatterns, sup)
        return finalPatterns, support, info, second

    def conditionalTransactions(self, condPatterns, support):
        """ It generates the conditional patterns with frequent items
            :param condPatterns: condPatterns generated from condition pattern method for respective node
            :type condPatterns: list
            :param support: the support of conditional pattern in tree
            :type support: list
        """
        global _minSup
        pat = []
        sup = []
        data1 = {}
        for i in range(len(condPatterns)):
            for j in condPatterns[i]:
                if j in data1:
                    data1[j] += support[i]
                else:
                    data1[j] = support[i]
        updatedDict = {}
        updatedDict = {k: v for k, v in data1.items() if v >= _minSup}
        count = 0
        for p in condPatterns:
            p1 = [v for v in p if v in updatedDict]
            trans = sorted(p1, key=lambda x: (updatedDict.get(x)), reverse=True)
            if len(trans) > 0:
                pat.append(trans)
                sup.append(support[count])
            count += 1
        return pat, sup, updatedDict

    def removeNode(self, nodeValue):
        """removing the node from tree
            :param nodeValue : it represents the node in tree
            :type nodeValue : node
        """
        for i in self.summaries[nodeValue]:
            del i.parent.children[nodeValue]

    def generatePatterns(self, prefix):
        """generates the patterns
            :param prefix : forms the combination of items
            :type prefix : list
        """
        global _finalPatterns, _minSup
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x))):
            pattern = prefix[:]
            pattern.append(i)
            s = 0
            for x in self.summaries[i]:
                if x.k <= 2:
                    s += x.probability
                elif x.k >= 3:
                    n = x.probability * pow(x.prefixProbability, (x.k - 2))
                    s += n
            _finalPatterns[tuple(pattern)] = self.info[i]
            if s >= _minSup:
                patterns, support, info, second = self.conditionalPatterns(i)
                conditionalTree = _Tree()
                conditionalTree.info = info.copy()
                for pat in range(len(patterns)):
                    conditionalTree.addConditionalTransaction(patterns[pat], support[pat], second[pat])
                if len(patterns) > 0:
                    conditionalTree.generatePatterns(pattern)
            self.removeNode(i)


class TubeP(_fp._frequentPatterns):
    """
    TubeP is one of the fastest algorithm to discover frequent patterns in a uncertain transactional database.
    Reference:
    --------
        Carson Kai-Sang LeungSyed,  Khairuzzaman Tanbeer, "Fast Tree-Based Mining of Frequent Itemsets from Uncertain Data",
         International Conference on Database Systems for Advanced Applications(DASFAA 2012), https://link.springer.com/chapter/10.1007/978-3-642-29038-1_21
    Attributes:
    ----------
        iFile : file
            Name of the Input file or path of the input file
        oFile : file
            Name of the output file or path of the output file
        minSup: float or int or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            To represent the total no of transaction
        tree : class
            To represents the Tree class
        itemSetCount : int
            To represents the total no of patterns
        finalPatterns : dict
            To store the complete patterns
    Methods:
    -------
        startMine()
            Mining process will start from here
        getPatterns()
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
        creatingItemSets(fileName)
            Scans the dataset and stores in a list format
        frequentOneItem()
            Extracts the one-length frequent patterns from database
        updateTransactions()
            Update the transactions by removing non-frequent items and sort the Database by item decreased support
        buildTree()
            After updating the Database, remaining items will be added into the tree by setting root node as null
        convert()
            to convert the user specified value
    Executing the code on terminal:
    -------
        Format:
        ------
        python3 TubeP.py <inputFile> <outputFile> <minSup>
        Examples:
        --------
        python3 TubeP.py sampleTDB.txt patterns.txt 3    (minSup  will be considered in support count or frequency)
    Sample run of importing the code:
    -------------------
        from PAMI.uncertainFrequentPattern.basic import tubeP as alg
        obj = alg.TubeP(iFile, minSup)
        obj.startMine()
        Patterns = obj.getPatterns()
        print("Total number of  Patterns:", len(Patterns))
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
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.\n
    """
    _startTime = float()
    _endTime = float()
    _minSup = str()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _rank = {}

    def __init__(self, iFile, minSup, sep='\t'):
        super().__init__(iFile, minSup, sep)

    def _creatingItemSets(self):
        """
        Scans the dataset and stores the transactions into Database variable
        """
        self._Database = []
        if isinstance(self._iFile, _fp._pd.DataFrame):
            uncertain, data = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
            if 'uncertain' in i:
                uncertain = self._iFile['uncertain'].tolist()
            for k in range(len(data)):
                tr = []
                for j in range(len(data[k])):
                    product = _Item(data[k][j], uncertain[k][j])
                    tr.append(product)
                self._Database.append(tr)

            # print(self.Database)
        if isinstance(self._iFile, str):
            if _fp._validators.url(self._iFile):
                data = _fp._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    tr = []
                    for i in temp:
                        i1 = i.index('(')
                        i2 = i.index(')')
                        item = i[0:i1]
                        probability = float(i[i1 + 1:i2])
                        product = _Item(item, probability)
                        tr.append(product)
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r') as f:
                        for line in f:
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            tr = []
                            for i in temp:
                                i1 = i.index('(')
                                i2 = i.index(')')
                                item = i[0:i1]
                                probability = float(i[i1 + 1:i2])
                                product = _Item(item, probability)
                                tr.append(product)
                            self._Database.append(tr)
                except IOError:
                    print("File Not Found")

    def _frequentOneItem(self):
        """takes the transactions and calculates the support of each item in the dataset and assign the
                    ranks to the items by decreasing support and returns the frequent items list
        """
        global _minSup
        mapSupport = {}
        for i in self._Database:
            for j in i:
                if j.item not in mapSupport:
                    mapSupport[j.item] = round(j.probability, 2)
                else:
                    mapSupport[j.item] += round(j.probability, 2)
        mapSupport = {k: round(v, 2) for k, v in mapSupport.items() if v >= self._minSup}
        plist = [k for k, v in sorted(mapSupport.items(), key=lambda x: x[1], reverse=True)]
        self._rank = dict([(index, item) for (item, index) in enumerate(plist)])
        return mapSupport, plist

    def _buildTree(self, data, info):
        """it takes the transactions and support of each item and construct the main tree with setting root
                    node as null
            :param data : it represents the one transactions in database
            :type data : list
            :param info : it represents the support of each item
            :type info : dictionary
        """
        rootNode = _Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            rootNode.addTransaction(data[i])
        return rootNode

    def _updateTransactions(self, dict1):
        """remove the items which are not frequent from transactions and updates the transactions with rank of items
            :param dict1 : frequent items with support
            :type dict1 : dictionary
        """
        list1 = []
        for tr in self._Database:
            list2 = []
            for i in range(0, len(tr)):
                if tr[i].item in dict1:
                    list2.append(tr[i])
            if len(list2) >= 2:
                basket = list2
                basket.sort(key=lambda val: self._rank[val.item])
                list2 = basket
                list1.append(list2)
        return list1

    def _Check(self, i, x):
        """To check the presence of item or pattern in transaction
            :param x: it represents the pattern
            :type x : list
            :param i : represents the uncertain transactions
            :type i : list
        """
        for m in x:
            k = 0
            for n in i:
                if m == n.item:
                    k += 1
            if k == 0:
                return 0
        return 1

    def _convert(self, value):
        """
            To convert the type of user specified minSup value
                :param value: user specified minSup value
                :return: converted type minSup value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self._Database) * value)
        if type(value) is str:
            if '.' in value:
                value = (len(self._Database) * value)
            else:
                value = int(value)
        return value

    def _removeFalsePositives(self):
        """
               To remove the false positive patterns generated in frequent patterns
               :return: patterns with accurate probability
        """
        global _finalPatterns
        periods = {}
        for i in self._Database:
            for x, y in _finalPatterns.items():
                if len(x) == 1:
                    periods[x] = y
                else:
                    s = 1
                    check = self._Check(i, x)
                    if check == 1:
                        for j in i:
                            if j.item in x:
                                s *= j.probability
                        if x in periods:
                            periods[x] += s
                        else:
                            periods[x] = s
        for x, y in periods.items():
            if y >= self._minSup:
                sample = str()
                for i in x:
                    sample = sample + i + " "
                self._finalPatterns[sample] = y

    def startMine(self):
        """Main method where the patterns are mined by constructing tree and remove the remove the false patterns
                           by counting the original support of a patterns
        """
        global _minSup
        self._startTime = _fp._time.time()
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        _minSup = self._minSup
        self._finalPatterns = {}
        mapSupport, plist = self._frequentOneItem()
        transactions1 = self._updateTransactions(mapSupport)
        info = {k: v for k, v in mapSupport.items()}
        Tree1 = self._buildTree(transactions1, info)
        Tree1.generatePatterns([])
        self._removeFalsePositives()
        print("Frequent patterns were generated successfully using TubeP algorithm")
        self._endTime = _fp._time.time()
        process = _fp._psutil.Process(_fp._os.getpid())
        self._memoryRSS = float()
        self._memoryUSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss

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

        dataframe = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataframe = _fp._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataframe

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns


if __name__ == "__main__":
    _ap = str()
    if len(_fp._sys.argv) == 4 or len(_fp._sys.argv) == 5:
        if len(_fp._sys.argv) == 5:
            _ap = TubeP(_fp._sys.argv[1], _fp._sys.argv[3], _fp._sys.argv[4])
        if len(_fp._sys.argv) == 4:
            _ap = TubeP(_fp._sys.argv[1], _fp._sys.argv[3])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of  Patterns:", len(_Patterns))
        _ap.savePatterns(_fp._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        '''l = [0.001, 0.004, 0.006]
        for i in l:
            _ap = TubeP('/Users/likhitha/Downloads/transactionalDb.csv', i, ' ')
            _ap.startMine()
            Patterns = _ap.getPatterns()
            print("Total number of Patterns:", len(Patterns))
            _ap.savePatterns('/Users/likhitha/Downloads/output.txt')
            memUSS = _ap.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = _ap.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = _ap.getRuntime()
            print("Total ExecutionTime in ms:", run)'''
        print("Error! The number of input parameters do not match the total number of parameters provided")