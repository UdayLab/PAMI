# BinaryFPGrowth discovers the complete set of frequent patterns directly from a binary (0/1) transactional dataset, where rows are transactions and columns are items. It accepts one-hot / boolean input as-is and extracts the transactions with a single vectorised pass, then stores them in a compressed FP-tree and mines the patterns from the tree without candidate generation.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.frequentPattern.basic import BinaryFPGrowth as alg
#
#             import numpy as np
#
#             iFile = np.array([[1, 0, 1, 1], [0, 1, 1, 0]])  # rows = transactions, cols = items
#
#             minSup = 1  # can also be specified between 0 and 1
#
#             obj = alg.BinaryFPGrowth(iFile, minSup)
#
#             obj.mine()
#
#             frequentPatterns = obj.getPatterns()
#
#             print("Total number of Frequent Patterns:", len(frequentPatterns))
#
#             obj.save(oFile)
#
#             Df = obj.getPatternsAsDataFrame()
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
Copyright (C)  2026 Rage Uday Kiran

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
from typing import Dict, List, Tuple, Union, Any
from deprecated import deprecated
from itertools import combinations
import numpy as _np

_fp._sys.setrecursionlimit(20000)


class _Node:
    """
    A class used to represent a node of the frequent-pattern tree.

    :**Attributes**:    - **item** -- *the item stored at this node.*
                        - **count** (*int*) -- *the support carried through this node.*
                        - **parent** (*_Node*) -- *the parent node.*
                        - **children** (*dict*) -- *child nodes keyed by item.*
    """

    def __init__(self, item, count, parent) -> None:
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

    def addChild(self, item, count=1) -> Any:
        """
        Adds (or increments) a child node for ``item`` and returns it.
        """
        if item not in self.children:
            self.children[item] = _Node(item, count, self)
        else:
            self.children[item].count += count
        return self.children[item]

    def traverse(self) -> Tuple[List, int]:
        """
        Walks up to the root and returns the prefix transaction (root-first) and this node's count.
        """
        transaction = []
        count = self.count
        node = self.parent
        while node.parent is not None:
            transaction.append(node.item)
            node = node.parent
        return transaction[::-1], count


class BinaryFPGrowth(_fp._frequentPatterns):
    """
    **About this algorithm**

    :**Description**:   BinaryFPGrowth discovers the complete set of frequent patterns directly from a *binary* (0/1) transactional
                        dataset in which rows represent transactions and columns represent items. It accepts one-hot / boolean input
                        as-is (no flattening to an item-token file) and extracts the present items of every transaction with a single
                        vectorised ``numpy.nonzero`` pass, while item supports are obtained from vectorised column sums. The
                        transactions are then stored in a compressed FP-tree and the patterns are mined from the tree using the
                        standard FP-Growth procedure, employing the downward-closure property to prune the search space.

    :**Reference**:  Han, J., Pei, J., Yin, Y. et al. Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern
                     Tree Approach. Data  Mining and Knowledge Discovery 8, 53-87 (2004). https://doi.org/10.1023

    :**Parameters**:    - **iFile** (*numpy.ndarray or pandas.DataFrame or str*) -- *The binary dataset. A 2D 0/1 (or boolean) NumPy array, a binary/one-hot pandas DataFrame (column labels are used as item names), or a path to a whitespace/sep-separated 0/1 matrix file with an optional header row of item names. A DataFrame holding a single ``'Transactions'`` column is also accepted and one-hot encoded automatically.*
                        - **oFile** (*str*) -- *Name of the output file to store complete set of frequent patterns.*
                        - **minSup** (*int or float or str*) -- *The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.*
                        - **sep** (*str*) -- *This variable is used to distinguish items from one another while reading a matrix file. The default separator is tab space. However, the users can override their default separator.*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the mining process.*
                        - **endTime** (*float*) -- *To record the completion time of the mining process.*
                        - **finalPatterns** (*dict*) -- *Storing the complete set of patterns in a dictionary variable.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*


    **Execution methods**

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 BinaryFPGrowth.py <inputFile> <outputFile> <minSup>

      Example Usage:

      (.venv) $ python3 BinaryFPGrowth.py binaryDB.txt patterns.txt 10.0

    .. note:: minSup can be specified  in support count or a value between 0 and 1.


    **Calling from a python program**

    .. code-block:: python

            from PAMI.frequentPattern.basic import BinaryFPGrowth as alg

            import numpy as np

            iFile = np.array([[1, 0, 1, 1], [0, 1, 1, 0]])  # rows = transactions, cols = items

            minSup = 1  # can also be specified between 0 and 1

            obj = alg.BinaryFPGrowth(iFile, minSup)

            obj.mine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.save(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


    **Credits**

    The complete program was written under the supervision of Professor Rage Uday Kiran.

    """

    _minSup = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()

    def __init__(self, iFile, minSup, sep="\t") -> None:
        super().__init__(iFile, minSup, sep)
        self._items: List[str] = []
        self._Database: List[List[str]] = []
        self._itemSupport: Dict[str, int] = {}
        self._nTransactions = 0

    def _peekHeader(self, path: str) -> bool:
        """
        Returns True if the first line of ``path`` contains non-numeric tokens (treated as a header of item names).
        """
        with open(path, 'r', encoding='utf-8') as f:
            first = f.readline().strip()
        if not first:
            return False
        delimiter = None if self._sep in ('\t', ' ', '') else self._sep
        tokens = first.split() if delimiter is None else first.split(delimiter)
        for tok in tokens:
            try:
                float(tok)
            except ValueError:
                return True
        return False

    def _matrixFromTransactions(self, transactions: List[List[str]]) -> Tuple[_np.ndarray, List[str]]:
        """
        One-hot encode item-token transactions into a boolean matrix and the sorted item vocabulary.
        """
        vocabulary = sorted({item for row in transactions for item in row})
        position = {item: idx for idx, item in enumerate(vocabulary)}
        matrix = _np.zeros((len(transactions), len(vocabulary)), dtype=bool)
        for rowIdx, row in enumerate(transactions):
            for item in row:
                matrix[rowIdx, position[item]] = True
        return matrix, vocabulary

    def _creatingItemSets(self) -> None:
        """
        Loads the binary dataset, then (a) extracts every transaction's present items with one vectorised
        ``numpy.nonzero`` pass into ``self._Database`` and (b) computes per-item supports with vectorised column sums.
        """
        matrix = None
        names = None

        if isinstance(self._iFile, _np.ndarray):
            matrix = self._iFile
            names = ["item" + str(j + 1) for j in range(matrix.shape[1])] if matrix.ndim == 2 else []

        elif isinstance(self._iFile, _fp._pd.DataFrame):
            if self._iFile.empty:
                self._items, self._Database, self._itemSupport, self._nTransactions = [], [], {}, 0
                return
            columns = self._iFile.columns.values.tolist()
            if 'Transactions' in columns:
                rows = [str(x).split(self._sep) for x in self._iFile['Transactions'].tolist()]
                rows = [[i for i in row if i] for row in rows]
                matrix, names = self._matrixFromTransactions(rows)
            else:
                matrix = self._iFile.to_numpy()
                names = [str(c) for c in columns]

        elif isinstance(self._iFile, str):
            delimiter = None if self._sep in ('\t', ' ', '') else self._sep
            if self._peekHeader(self._iFile):
                with open(self._iFile, 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
                names = header.split() if delimiter is None else header.split(delimiter)
                matrix = _np.loadtxt(self._iFile, delimiter=delimiter, skiprows=1, ndmin=2)
            else:
                matrix = _np.loadtxt(self._iFile, delimiter=delimiter, ndmin=2)
                names = ["item" + str(j + 1) for j in range(matrix.shape[1])]
        else:
            raise TypeError("iFile must be a numpy.ndarray, a pandas.DataFrame or a path to a binary matrix file")

        matrix = _np.asarray(matrix)
        if matrix.ndim != 2 or matrix.size == 0:
            self._items, self._Database, self._itemSupport, self._nTransactions = [], [], {}, 0
            return

        boolMatrix = matrix.astype(bool)
        self._items = [str(n) for n in names]
        self._nTransactions = boolMatrix.shape[0]

        nameArray = _np.asarray(self._items, dtype=object)
        # Vectorised per-item support from column sums.
        columnSums = boolMatrix.sum(axis=0)
        self._itemSupport = {self._items[j]: int(columnSums[j]) for j in range(len(self._items))}
        # Single vectorised pass to split present-item ids per transaction.
        rows, cols = _np.nonzero(boolMatrix)
        tokens = nameArray[cols]
        counts = _np.bincount(rows, minlength=self._nTransactions)
        boundaries = _np.cumsum(counts)[:-1]
        self._Database = [list(group) for group in _np.split(tokens, boundaries)]

    def _convert(self, value: Union[int, float, str]) -> Union[int, float]:
        """
        To convert the user specified minSup value into an absolute support count.

        :param value: user specified minSup value
        :type value: int or float or str
        :return: converted minSup
        :rtype: int or float
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self._nTransactions * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self._nTransactions * value)
            else:
                value = int(value)
        return value

    def _construct(self, items, data, minSup):
        """
        Constructs the FP-tree from the given transactions (standard FP-Growth construction).

        :param items: dict mapping item -> frequency.
        :param data: list of transactions.
        :param minSup: minimum support threshold.
        :return: (root node, dict item -> [set(nodes), support]).
        """
        items = {k: v for k, v in items.items() if v >= minSup}

        root = _Node([], 0, None)
        itemNodes = {}
        for line in data:
            currNode = root
            line = sorted([item for item in line if item in items], key=lambda x: items[x], reverse=True)
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
        Generates all non-empty combinations of the items of a (single) conditional prefix path.
        """
        all_combinations_list = []
        for r in range(1, len(arr) + 1):
            all_combinations_list.extend(combinations(arr, r))
        return all_combinations_list

    def _recursive(self, root, itemNode, minSup, patterns):
        """
        Recursively mines the FP-tree to emit every frequent pattern (standard FP-Growth recursion).
        """
        itemNode = {k: v for k, v in sorted(itemNode.items(), key=lambda x: x[1][1])}

        for item in itemNode:
            if itemNode[item][1] < self._minSup:
                break

            newRoot = _Node(root.item + [item], 0, None)
            self._finalPatterns[tuple(newRoot.item)] = itemNode[item][1]
            newItemNode = {}

            if len(itemNode[item][0]) == 1:
                transaction, count = itemNode[item][0].pop().traverse()
                if len(transaction) == 0:
                    continue
                for comb in self._all_combinations(transaction):
                    self._finalPatterns[tuple(list(comb) + newRoot.item)] = count

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

            itemCount = {k: v for k, v in itemCount.items() if v >= minSup}
            if len(itemCount) == 0:
                continue

            for transaction, count in transactions.items():
                transaction = sorted([it for it in transaction if it in itemCount], key=lambda x: itemCount[x], reverse=True)
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

            self._recursive(newRoot, newItemNode, minSup, patterns)

    def mine(self) -> None:
        """
        Frequent pattern mining process will start from here.
        """
        self._startTime = _fp._time.time()
        self._finalPatterns = {}
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)

        if self._nTransactions > 0 and self._itemSupport:
            root, itemNode = self._construct(self._itemSupport, self._Database, self._minSup)
            self._recursive(root, itemNode, self._minSup, self._finalPatterns)

        self._endTime = _fp._time.time()
        process = _fp._psutil.Process(_fp._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using BinaryFPGrowth algorithm ")

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """
        Frequent pattern mining process will start from here
        """
        self.mine()

    def getMemoryUSS(self) -> float:
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """
        return self._memoryUSS

    def getMemoryRSS(self) -> float:
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """
        return self._memoryRSS

    def getRuntime(self) -> float:
        """
        Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """
        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self) -> _fp._pd.DataFrame:
        """
        Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataFrame = _fp._pd.DataFrame(list([[self._sep.join(x), y] for x, y in self._finalPatterns.items()]),
                                      columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, oFile: str, seperator="\t") -> None:
        """
        Complete set of frequent patterns will be loaded in to an output file

        :param oFile: name of the output file
        :type oFile: csvfile
        :param seperator: variable to store the separator
        :type seperator: string
        :return: None
        """
        with open(oFile, 'w') as f:
            for x, y in self._finalPatterns.items():
                x = seperator.join(x)
                f.write(f"{x}:{y}\n")

    def getPatterns(self) -> Dict[Tuple[str, ...], int]:
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
            _ap = BinaryFPGrowth(_fp._sys.argv[1], _fp._sys.argv[3], _fp._sys.argv[4])
        if len(_fp._sys.argv) == 4:
            _ap = BinaryFPGrowth(_fp._sys.argv[1], _fp._sys.argv[3])
        _ap.mine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_fp._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
