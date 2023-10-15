# UPFPGrowthPlus is used to discover periodic-frequent patterns in an uncertain temporal database.
#
#    **Importing this algorithm into a python program**
#    --------------------------------------------------------
#
#
#     from PAMI.uncertainPeriodicFrequentPattern.basic import UPFPGrowthPlus as alg
#
#     obj = alg.UPFPGrowthPlus(iFile, minSup, maxPer)
#
#     obj.startMine()
#
#     periodicFrequentPatterns = obj.getPatterns()
#
#     print("Total number of uncertain Periodic Frequent Patterns:", len(periodicFrequentPatterns))
#
#     obj.save(oFile)
#
#     Df = obj.getPatternsAsDataFrame()
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


from PAMI.uncertainPeriodicFrequentPattern.basic import abstract as _ab

_minSup = float()
_maxPer = float()
_lno = int()
_first = int()
_last = int()


class _Item:
    """
    A class used to represent the item with probability in transaction of dataset

    Attributes:
    __________
        item : int or string
            Represents the name of the item
        probability : float
            Represent the existential probability(likelihood presence) of an item
    """

    def __init__(self, item, probability):
        self.item = item
        self.probability = probability


def printTree(root):
    """
    To print the tree with nodes with item name, probability, timestamps, and second probability respectively.
    ...
    Attributes:
    ----------
       :param root: Node
       :return: print all Tree with nodes with items, probability, parent item, timestamps, second probability respectively.
    """
    for x, y in root.children.items():
        print(x, y.item, y.probability, y.parent.item, y.tids, y.secondProbability)
        printTree(y)


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
        self.secondProbability = 1
        self.p = 1
        self.children = children
        self.parent = None
        self.TimeStamps = []

    def addChild(self, node):
        """
            to add children details to parent node

            :param node: children node

            :return: update parent node children
        """
        self.children[node.item] = node
        node.parent = self


class _Tree(object):
    """
        A class used to represent the frequentPatternGrowth tree structure

        ...

        Attributes:
        ----------
            root: Node
                Represents the root node of the tree
            summaries: dictionary
                storing the nodes with same item name
            info: dictionary
                stores the support of items


        Methods:
        -------
            addTransaction(transaction)
                creating transaction as a branch in Tree
            addConditionalTransaction(prefixPaths, supportOfItems)
                construct the conditional tree for prefix paths
            getConditionalPatterns(Node)
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


    def addTransaction(self, transaction, tid):
        """
            adding transaction into tree

            :param transaction : it represents the one transaction in database

            :type transaction : list

            :param tid : the timestamp of transaction

            :type tid : list
        """
        currentNode = self.root
        k = 0
        for i in range(len(transaction)):
            k += 1
            if transaction[i].item not in currentNode.children:
                newNode = _Node(transaction[i].item, {})
                newNode.k = k
                newNode.secondProbability = transaction[i].probability
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
                currentNode.secondProbability = max(transaction[i].probability, currentNode.secondProbability)
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
        currentNode.TimeStamps = currentNode.TimeStamps + tid

    def addConditionalPatterns(self, transaction, tid, sup, probability):
        """
            constructing conditional tree from prefixPaths

            :param transaction : it represents the one transaction in database

            :type transaction : list

            :param tid : timestamps of a pattern or transaction in tree

            :param tid : list

            :param sup : support of prefixPath taken at last child of the path

            :type sup : int
        """
        currentNode = self.root
        k = 0
        for i in range(len(transaction)):
            k += 1
            if transaction[i] not in currentNode.children:
                newNode = _Node(transaction[i], {})
                newNode.k = k
                newNode.probability = sup
                newNode.secondProbability = probability
                currentNode.addChild(newNode)
                if transaction[i] in self.summaries:
                    self.summaries[transaction[i]].append(newNode)
                else:
                    self.summaries[transaction[i]] = [newNode]
                currentNode = newNode
            else:
                currentNode = currentNode.children[transaction[i]]
                currentNode.k = k
                currentNode.probability += sup
                currentNode.secondProbability = max(probability, currentNode.secondProbability)
        currentNode.TimeStamps = currentNode.TimeStamps + tid

    def conditionalPatterns(self, alpha):
        """generates all the conditional patterns of respective node

                :param alpha : it represents the Node in tree

                :type alpha : Node
        """
        finalPatterns = []
        finalSets = []
        sup = []
        prob = []
        for i in self.summaries[alpha]:
            set1 = i.TimeStamps
            s = i.probability
            p = i.secondProbability
            set2 = []
            while i.parent.item is not None:
                set2.append(i.parent.item)
                i = i.parent
            if len(set2) > 0:
                set2.reverse()
                finalPatterns.append(set2)
                finalSets.append(set1)
                sup.append(s)
                prob.append(p)
        finalPatterns, finalSets, support, prob, info = self.conditionalTransactions(finalPatterns, finalSets, sup, prob)
        return finalPatterns, finalSets, support, prob, info

    def removeNode(self, nodeValue):
        """removing the node from tree

            :param nodeValue : it represents the node in tree

            :type nodeValue : node
        """
        for i in self.summaries[nodeValue]:
            i.parent.TimeStamps = i.parent.TimeStamps + i.TimeStamps
            del i.parent.children[nodeValue]

    def getPeriodAndSupport(self, support, TimeStamps):
        """
        To calculate the periodicity of given timestamps

        Parameters
        ----------
        support: support of pattern
        TimeStamps: timmeStamps of a pattern

        Returns
        -------
        support and period

        """
        global _maxPer
        global _lno
        TimeStamps.sort()
        cur = 0
        per = 0
        sup = support
        for j in range(len(TimeStamps)):
            per = max(per, TimeStamps[j] - cur)
            if per > _maxPer:
                return [0, 0]
            cur = TimeStamps[j]
        per = max(per, _lno - cur)
        return [sup, per]

    def conditionalTransactions(self, conditionalPatterns, conditionalTimeStamps, support, probability):
        """ It generates the conditional patterns with frequent items

            :param conditionalPatterns : conditional patterns generated from conditionalPatterns() method for respective node

            :type conditionalPatterns : list

            :param conditionalTimeStamps : timestamps of respective conditional timestamps

            :type conditionalTimeStamps : list

            :param support : the support of conditional pattern in tree

            :type support : list
        """
        global _minSup, _maxPer, _lno
        pat = []
        TimeStamps = []
        sup = []
        prob = []
        data1 = {}
        count = {}
        for i in range(len(conditionalPatterns)):
            for j in conditionalPatterns[i]:
                if j in data1:
                    data1[j] = data1[j] + conditionalTimeStamps[i]
                    count[j] += support[i]
                else:
                    data1[j] = conditionalTimeStamps[i]
                    count[j] = support[i]
        updatedDict = {}
        for m in data1:
            updatedDict[m] = self.getPeriodAndSupport(count[m], data1[m])
        updatedDict = {k: v for k, v in updatedDict.items() if v[0] >= _minSup and v[1] <= _maxPer}
        count = 0
        for p in conditionalPatterns:
            p1 = [v for v in p if v in updatedDict]
            trans = sorted(p1, key=lambda x: (updatedDict.get(x)[0]), reverse=True)
            if len(trans) > 0:
                pat.append(trans)
                TimeStamps.append(conditionalTimeStamps[count])
                sup.append(support[count])
                prob.append(probability[count])
            count += 1
        return pat, TimeStamps, sup, prob, updatedDict

    def generatePatterns(self, prefix, periodic):
        """generates the patterns

            :param prefix : forms the combination of items

            :type prefix : list
        """
        global _minSup
        for i in sorted(self.summaries, key=lambda x: (self.info.get(x)[0])):
            pattern = prefix[:]
            pattern.append(i)
            s = 0
            secProb = []
            kk = int()
            for x in self.summaries[i]:
                if x.k <= 2:
                    s += x.probability
                elif x.k >= 3:
                    n = x.probability * pow(x.secondProbability, (x.k - 2))
                    s += n
            periodic[tuple(pattern)] = self.info[i]
            periodic[tuple(pattern)] = self.info[i]
            if s >= _minSup:
                periodic[tuple(pattern)] = self.info[i]
                patterns, TimeStamps, support, probability, info = self.conditionalPatterns(i)
                conditionalTree = _Tree()
                conditionalTree.info = info.copy()
                for pat in range(len(patterns)):
                    conditionalTree.addConditionalPatterns(patterns[pat], TimeStamps[pat], support[pat], probability[pat])
                if len(patterns) > 0:
                    conditionalTree.generatePatterns(pattern, periodic)
            self.removeNode(i)

class UPFPGrowthPlus(_ab._periodicFrequentPatterns):
    """
    :Description:

        basic Plus is  to discover periodic-frequent patterns in a uncertain temporal database.

    :Reference:

          Palla Likhitha, Rage Veena,Rage Uday Kiran, Koji Zettsu, Masashi Toyoda, Philippe Fournier-Viger, (2023). 
          UPFP-growth++: An Efficient Algorithm to Find Periodic-Frequent Patterns in Uncertain Temporal Databases. 
          ICONIP 2022. Communications in Computer and Information Science, vol 1792. Springer, Singapore.
          https://doi.org/10.1007/978-981-99-1642-9_16

    Attributes:

        iFile: file
            Name of the Input file or path of input file
        oFile: file
            Name of the output file or path of output file
        minSup: int or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        maxPer: int or float or str
            The user can specify maxPer either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        sep: str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        memoryUSS: float
            To store the total amount of USS memory consumed by the program
        memoryRSS: float
            To store the total amount of RSS memory consumed by the program
        startTime: float
            To record the start time of the mining process
        endTime: float
            To record the completion time of the mining process
        Database: list
            To store the transactions of a database in list
        mapSupport: Dictionary
            To maintain the information of item and their frequency
        lno: int
            To represent the total no of transaction
        tree: class
            To represents the Tree class
        itemSetCount: int
            To represents the total no of patterns
        finalPatterns: dict
            To store the complete patterns
    Methods:
    ---------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingItemSets(fileName)
            Scans the dataset and stores in a list format
        updateDatabases()
            Update the database by removing aperiodic items and sort the Database by item decreased support
        buildTree()
            After updating the Database, remaining items will be added into the tree by setting root node as null
        convert()
            to convert the user specified value
        PeriodicFrequentOneItems()
            To extract the one-length periodic-frequent items

    Executing the code on terminal:
    --------------------------------------------
        Format:
        -------
           >>> python3 UPFPGrowthPlus.py <inputFile> <outputFile> <minSup> <maxPer>
        Examples:
        ---------
           >>> python3 UPFPGrowthPlus.py sampleTDB.txt patterns.txt 0.3 4     (minSup and maxPer will be considered in support count or frequency)

     Importing this algorithm into a python program
    -----------------------------------------------------------------

    .. code-block:: python

            from PAMI.uncertainPeriodicFrequentPattern import UPFPGrowthPlus as alg

            obj = alg.UPFPGrowthPlus(iFile, minSup, maxPer)

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of uncertain Periodic Frequent Patterns:", len(periodicFrequentPatterns))

            obj.save(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)

    Credits:
    -----------
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.\n

        """
    _startTime = float()
    _endTime = float()
    _minSup = float()
    _maxPer = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _rank = {}
    _lno = 0
    _periodic = {}

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """

        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            uncertain, data, ts = [], [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self._iFile['Transactions'].tolist()
            if 'uncertain' in i:
                uncertain = self._iFile['uncertain'].tolist()
            for k in range(len(data)):
                tr = [ts[k]]
                for j in range(len(k)):
                    product = _Item(data[k][j], uncertain[k][j])
                    tr.append(product)
                self._Database.append(tr)
                self._lno += 1

            # print(self.Database)
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.decode("utf-8")
                    line = line.strip()
                    line = [i for i in line.split(':')]
                    temp1 = [i.rstrip() for i in line[0].split(self._sep)]
                    temp2 = [i.rstrip() for i in line[1].split(self._sep)]
                    temp1 = [x for x in temp1 if x]
                    temp2 = [x for x in temp2 if x]
                    tr = [int(temp1[0])]
                    for i in range(len(temp1[1:])):
                        item = temp1[i]
                        probability = float(temp2[i])
                        product = _Item(item, probability)
                        tr.append(product)
                    self._lno += 1
                    self._Database.append(tr)
            else:
                try:
                    count = 0
                    with open(self._iFile, 'r') as f:
                        for line in f:
                            line = line.strip()
                            line = [i for i in line.split(':')]
                            temp1 = [i.rstrip() for i in line[0].split(self._sep)]
                            temp2 = [i.rstrip() for i in line[1].split(self._sep)]
                            temp1 = [x for x in temp1 if x]
                            temp2 = [x for x in temp2 if x]
                            tr = [int(temp1[0])]
                            for i in range(len(temp1[1:])):
                                item = temp1[i]
                                probability = float(temp2[i])
                                product = _Item(item, probability)
                                tr.append(product)
                            self._lno += 1
                            self._Database.append(tr)
                except IOError:
                    print("File Not Found")

    def _PeriodicFrequentOneItems(self):
        """takes the transactions and calculates the support of each item in the dataset and assign the
                            ranks to the items by decreasing support and returns the frequent items list

        """
        global first, last
        mapSupport = {}
        for i in self._Database:
            n = int(i[0])
            for j in i[1:]:
                if j.item not in mapSupport:
                    mapSupport[j.item] = [round(j.probability, 3), abs(0 - n), n]
                else:
                    mapSupport[j.item][0] += round(j.probability, 2)
                    mapSupport[j.item][1] = max(mapSupport[j.item][1], abs(n - mapSupport[j.item][2]))
                    mapSupport[j.item][2] = n
        for key in mapSupport:
            mapSupport[key][1] = max(mapSupport[key][1], self._lno - mapSupport[key][2])
        mapSupport = {k: [round(v[0], 2), v[1]] for k, v in mapSupport.items() if
                      v[1] <= self._maxPer and v[0] >= self._minSup}
        plist = [k for k, v in sorted(mapSupport.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        self._rank = dict([(index, item) for (item, index) in enumerate(plist)])
        return mapSupport, plist

    def _buildTree(self, data, info):
        """it takes the transactions and support of each item and construct the main tree with setting root
                            node as null

            :param data : it represents the one transaction in database

            :type data : list

            :param info : it represents the support of each item

            :type info : dictionary
        """
        rootNode = _Tree()
        rootNode.info = info.copy()
        for i in range(len(data)):
            set1 = [data[i][0]]
            rootNode.addTransaction(data[i][1:], set1)
            #printTree(rootNode)
            #print("....")
        return rootNode

    def _updateTransactions(self, dict1):
        """remove the items which are not frequent from transactions and updates the transactions with rank of items

                :param dict1 : frequent items with support

                :type dict1 : dictionary
        """
        list1 = []
        for tr in self._Database:
            list2 = [int(tr[0])]
            for i in range(1, len(tr)):
                if tr[i].item in dict1:
                    list2.append(tr[i])
            if len(list2) >= 2:
                basket = list2[1:]
                basket.sort(key=lambda val: self._rank[val.item])
                list2[1:] = basket[0:]
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
            To convert the given user specified value

            :param value: user specified value

            :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = float(value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        return value

    def _removeFalsePositives(self):
        """
        To remove false positives in generated patterns

        :return: original patterns
        """
        periods = {}
        for i in self._Database:
            for x, y in self._periodic.items():
                if len(x) == 1:
                    periods[x] = y
                else:
                    s = 1
                    check = self._Check(i[1:], x)
                    if check == 1:
                        for j in i[1:]:
                            if j.item in x:
                                s *= j.probability
                        if x in periods:
                            periods[x][0] += s
                        else:
                            periods[x] = [s, y[1]]
        count = 0
        for x, y in periods.items():
            if y[0] >= _minSup:
                count += 1
                sample = str()
                for i in x:
                    sample = sample + i + " "
                self._finalPatterns[sample] = y
        #print("Total false patterns generated:", len(self._periodic) - count)

    def startMine(self):
        """Main method where the patterns are mined by constructing tree and remove the false patterns
                           by counting the original support of a patterns


               """
        global _minSup, _maxPer, _first, _last, _lno
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._maxPer = self._convert(self._maxPer)
        self._finalPatterns = {}
        _minSup, _maxPer, _lno = self._minSup, self._maxPer, len(self._Database)
        mapSupport, plist = self._PeriodicFrequentOneItems()
        updatedTrans = self._updateTransactions(mapSupport)
        info = {k: v for k, v in mapSupport.items()}
        root = self._buildTree(updatedTrans, info)
        self._periodic = {}
        root.generatePatterns([], self._periodic)
        self._removeFalsePositives()
        print("Periodic Frequent patterns were generated successfully using UPFP-Growth++ algorithm")
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
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
        """ Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataframe = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataframe

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to an output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """ This function is used to print the results
        """
        print("Total number of  Uncertain Periodic-Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = UPFPGrowthPlus(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = UPFPGrowthPlus(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of Patterns:", len(_Patterns))
        _ap.savePatterns(_ab._sys.argv[2])
        # print(ap.getPatternsAsDataFrame())
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
