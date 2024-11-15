# FPGrowth is one of the fundamental algorithm to discover frequent patterns in a transactional database. It stores the database in compressed fp-tree decreasing the memory usage and extracts the patterns from tree.It  employs downward closure property to  reduce the search space effectively.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.frequentPattern.basic import FPGrowth as alg
#
#             iFile = 'sampleDB.txt'
#
#             minSup = 10  # can also be specified between 0 and 1
#
#             obj = alg.FPGrowth(iFile, minSup)
#
#             obj.mine()
#
#             frequentPatterns = obj.getPatterns()
#
#             print("Total number of Frequent Patterns:", len(frequentPatterns))
#
#             obj.save(oFile)
#
#             Df = obj.getPatternInDataFrame()
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime()
#
#             print("Total ExecutionTime in seconds:", run)
#


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
"""

from PAMI.frequentPattern.basic import abstract as _fp
from typing import List, Dict, Tuple, Any
from deprecated import deprecated
from itertools import combinations
from collections import Counter

_minSup = str()
_fp._sys.setrecursionlimit(20000)


class _Node:
    """
    A class used to represent the node of frequentPatternTree

    :**Attributes**:    - **itemId** (*int*) -- *storing item of a node.*
                        - **counter** (*int*) -- *To maintain the support of node.*
                        - **parent** (*node*) -- *To maintain the parent of node.*
                        - **children** (*list*) -- *To maintain the children of node.*

    :**Methods**:   - **addChild(node)** -- *Updates the nodes children list and parent for the given node.*
    """

    def __init__(self, item, count, parent) -> None:
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

    def addChild(self, item, count = 1) -> Any:
        """

        Adds a child node to the current node with the specified item and count.

        :param item: The item associated with the child node.
        :type item: List
        :param count: The count or support of the item. Default is 1.
        :type count: int
        :return: The child node added.
        :rtype: List
        """
        if item not in self.children:
            self.children[item] = _Node(item, count, self)
        else:
            self.children[item].count += count
        return self.children[item]
    
    def traverse(self) -> Tuple[List[int], int]:
        """
        Traversing the tree to get the transaction

        :return: transaction and count of each item in transaction
        :rtype: Tuple, List and int
        """
        transaction = []
        count = self.count
        node = self.parent
        while node.parent is not None:
            transaction.append(node.item)
            node = node.parent
        return transaction[::-1], count


class FPGrowth(_fp._frequentPatterns):
    """
    **About this algorithm**

    :**Description**:   FPGrowth is one of the fundamental algorithm to discover frequent patterns in a transactional database. It stores the database in compressed fp-tree decreasing the memory usage and extracts the patterns from tree.It employs downward closure property to  reduce the search space effectively.

    :**Reference**:  Han, J., Pei, J., Yin, Y. et al. Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern
                     Tree Approach. Data  Mining and Knowledge Discovery 8, 53–87 (2004). https://doi.org/10.1023

    :**Parameters**:    - **iFile** (*str or URL or dataFrame*) -- *Name of the Input file to mine complete set of frequent patterns.*
                        - **oFile** (*str*) -- *Name of the output file to store complete set of frequent patterns.*
                        - **minSup** (*int or float or str*) -- *The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.*
                        - **sep** (*str*) -- *This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the mining process.*
                        - **endTime** (*float*) -- *To record the completion time of the mining process.*
                        - **finalPatterns** (*dict*) -- *Storing the complete set of patterns in a dictionary variable.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*
                        - **Database** (*list*) -- *To store the transactions of a database in list.*
                        - **mapSupport** (*Dictionary*) -- *To maintain the information of item and their frequency.*
                        - **tree** (*class*) --  *it represents the Tree class.*


    **Execution methods**

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 FPGrowth.py <inputFile> <outputFile> <minSup>

      Example Usage:

      (.venv) $ python3 FPGrowth.py sampleDB.txt patterns.txt 10.0

    .. note:: minSup can be specified  in support count or a value between 0 and 1.


    **Calling from a python program**

    .. code-block:: python

            from PAMI.frequentPattern.basic import FPGrowth as alg

            iFile = 'sampleDB.txt'

            minSup = 10  # can also be specified between 0 and 1

            obj = alg.FPGrowth(iFile, minSup)

            obj.mine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.savePatterns(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


    **Credits:**

    The complete program was written by P. Likhitha and revised by Tarun Sreepada under the supervision of Professor Rage Uday Kiran.

    """

    __startTime = float()
    __endTime = float()
    _minSup = str()
    __finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    __memoryUSS = float()
    __memoryRSS = float()
    __Database = []
    __mapSupport = {}
    __lno = 0
    __rank = {}
    __rankDup = {}

    def __init__(self, iFile, minSup, sep='\t') -> None:
        super().__init__(iFile, minSup, sep)

    def __creatingItemSets(self) -> None:
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self.__Database = []
        if isinstance(self._iFile, _fp._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self.__Database = self._iFile['Transactions'].tolist()
                self.__Database = [x.split(self._sep) for x in self.__Database]
            else:
                print("The column name should be Transactions and each line should be separated by tab space or a seperator specified by the user")
                

            #print(self.Database)
        if isinstance(self._iFile, str):
            if _fp._validators.url(self._iFile):
                data = _fp._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self.__Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self.__Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def __convert(self, value) -> float:
        """

        To convert the type of user specified minSup value

        :param value: user specified minSup value
        :return: converted type
        :rtype: float
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self.__Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self.__Database) * value)
            else:
                value = int(value)
        return value
    
    def _construct(self, items, data, minSup):
        """
        Constructs the FP-tree from the given transactions.

        :param items: A dictionary containing item frequencies.
        :type items: Dict
        :param data: A list of transactions.
        :type data: List
        :param minSup: The minimum support threshold.
        :type minSup: int
        :return: The root node of the constructed FP-tree and a dictionary containing information about nodes associated with each item.
        :rtype: Tuple[_Node, Dict]
        """

        items = {k: v for k, v in items.items() if v >= minSup}

        root = _Node([], 0, None)
        itemNodes = {}
        for line in data:
            currNode = root
            line = sorted([item for item in line if item in items], key = lambda x: items[x], reverse = True)
            for item in line:
                currNode = currNode.addChild(item)
                if item in itemNodes:
                    itemNodes[item][0].add(currNode)
                    itemNodes[item][1] += 1
                else:
                    itemNodes[item] = [set([currNode]), 1]

        return root, itemNodes

    def _all_combinations(self, arr):
        """

        Generates all possible combinations of items from a given transaction.

        :param arr: A list of items in a transaction.
        :type arr: List
        :return: A list containing all possible combinations of items.
        :rtype: List
        """

        all_combinations_list = []
        for r in range(1, len(arr) + 1):
            all_combinations_list.extend(combinations(arr, r))
        return all_combinations_list
    
    def _recursive(self, root, itemNode, minSup, patterns):
        """

         Recursively explores the FP-tree to generate frequent patterns.

         :param root: The root node of the current subtree.
         :type root: _Node
         :param itemNode: A dictionary containing information about the nodes associated with each item.
         :type itemNode: Dict
         :param minSup: The minimum support threshold.
         :type minSup: int
         :param patterns: A dictionary to store the generated frequent patterns.
         :type patterns: Dict
        """
        itemNode = {k: v for k, v in sorted(itemNode.items(), key = lambda x: x[1][1])}

        for item in itemNode:
            if itemNode[item][1] < self._minSup:
                break 

            newRoot = _Node(root.item + [item], 0, None)
            # pat = "\t".join([str(i) for i in newRoot.item])
            # self.__finalPatterns[pat] = itemNode[item][1]
            self._finalPatterns[tuple(newRoot.item)] = itemNode[item][1]
            newItemNode = {}

            if len(itemNode[item][0]) == 1:
                transaction, count = itemNode[item][0].pop().traverse()
                if len(transaction) == 0:
                    continue
                combination = self._all_combinations(transaction)
                for comb in combination:
                    # pat = "\t".join([str(i) for i in comb])
                    # pat = pat + "\t" + "\t".join([str(i) for i in newRoot.item])
                    # self.__finalPatterns[pat] = count
                    self._finalPatterns[tuple(list(comb) + newRoot.item)] = count
                pass


            itemCount = {}
            transactions = {}
            for node in itemNode[item][0]:
                transaction, count = node.traverse()
                if len(transaction) == 0:
                    continue
                if tuple(transaction) in transactions:
                    transactions[tuple(transaction)] += count
                else:
                    transactions[tuple(transaction)] = count


                for transaction_item in transaction:
                    if transaction_item in itemCount:
                        itemCount[transaction_item] += count
                    else:
                        itemCount[transaction_item] = count


            # remove items that are below minSup
            itemCount = {k: v for k, v in itemCount.items() if v >= minSup}
            if len(itemCount) == 0:
                continue

            for transaction, count in transactions.items():
                transaction = sorted([item for item in transaction if item in itemCount], key = lambda x: itemCount[x], reverse = True)
                currNode = newRoot
                for item_ in transaction:
                    currNode = currNode.addChild(item_, count)
                    if item_ in newItemNode:
                        newItemNode[item_][0].add(currNode)
                        newItemNode[item_][1] += count
                    else:
                        newItemNode[item_] = [set([currNode]), count]

            if len(newItemNode) < 1:
                continue

            # mine(newRoot, newItemNode, minSup, patterns)
            self._recursive(newRoot, newItemNode, minSup, patterns)


    def mine(self) -> None:
        """
        Main program to start the operation
        """
        global _minSup
        self.__startTime = _fp._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        self.__creatingItemSets()
        self._minSup = self.__convert(self._minSup)
        _minSup = self._minSup

        itemCount = Counter()
        for line in self.__Database:
            itemCount.update(line)

        root, itemNode = self._construct(itemCount, self.__Database, self._minSup)
        self._recursive(root, itemNode, self._minSup, self.__finalPatterns)
        
        print("Frequent patterns were generated successfully using frequentPatternGrowth algorithm")
        self.__endTime = _fp._time.time()
        self.__memoryUSS = float()
        self.__memoryRSS = float()
        process = _fp._psutil.Process(_fp._os.getpid())
        self.__memoryUSS = process.memory_full_info().uss
        self.__memoryRSS = process.memory_info().rss

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self):
        """
        Starting the mining process
        """
        self.mine()

    def getMemoryUSS(self) -> float:
        """

        Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self.__memoryUSS

    def getMemoryRSS(self) -> float:
        """

        Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self.__memoryRSS

    def getRuntime(self) -> float:
        """

        Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self.__endTime - self.__startTime
    

    def getPatternsAsDataFrame(self) -> _fp._pd.DataFrame:
        """

        Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        # # dataframe = {}
        # # data = []
        # # for a, b in self.__finalPatterns.items():
        # #     data.append([a.replace('\t', ' '), b])
        # #     dataframe = _fp._pd.DataFrame(data, columns=['Patterns', 'Support'])
        # dataFrame = _fp._pd.DataFrame(list(self._finalPatterns.items()), columns=['Patterns', 'Support'])
        dataFrame = _fp._pd.DataFrame(list([[" ".join(x), y] for x,y in self._finalPatterns.items()]), columns=['Patterns', 'Support'])

        return dataFrame

    def save(self, outFile: str, seperator = "\t" ) -> None:
        """

        Complete set of frequent patterns will be loaded in to an output file

        :param outFile: name of the output file
        :type outFile: csvfile
        :param seperator: variable to store the separator
        :type seperator: string
        :return: None
        """
        with open(outFile, 'w') as f:
            for x, y in self._finalPatterns.items():
                x = seperator.join(x)
                f.write(f"{x}:{y}\n")

    def getPatterns(self) -> Dict[str, int]:
        """

        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns
    
    def printResults(self) -> None:
        """
        This function is used to print the results
        """
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_fp._sys.argv) == 4 or len(_fp._sys.argv) == 5:
        if len(_fp._sys.argv) == 5:
            _ap = FPGrowth(_fp._sys.argv[1], _fp._sys.argv[3], _fp._sys.argv[4])
        if len(_fp._sys.argv) == 4:
            _ap = FPGrowth(_fp._sys.argv[1], _fp._sys.argv[3])
        _ap.startMine()
        _ap.mine()
        print("Total number of Frequent Patterns:", len( _ap.getPatterns()))
        _ap.save(_fp._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
