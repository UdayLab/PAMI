# BinaryECLAT discovers the complete set of frequent patterns directly from a binary (0/1) transactional dataset, where rows are transactions and columns are items. ECLAT is a vertical (tidset) algorithm, so a binary matrix is already its natural data layout: each item column is packed into a bitset and itemsets are extended depth-first by intersecting bitsets with a vectorised pop-count.
#
# **Importing this algorithm into a python program**
#
#             import PAMI.frequentPattern.basic.BinaryECLAT as alg
#
#             import numpy as np
#
#             iFile = np.array([[1, 0, 1, 1], [0, 1, 1, 0]])  # rows = transactions, cols = items
#
#             minSup = 1  # can also be specified between 0 and 1
#
#             obj = alg.BinaryECLAT(iFile, minSup)
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

from PAMI.frequentPattern.basic import abstract as _ab
from typing import Dict, List, Tuple, Union
from deprecated import deprecated
import numpy as _np

_ab._sys.setrecursionlimit(20000)


class BinaryECLAT(_ab._frequentPatterns):
    """
    **About this algorithm**

    :**Description**: BinaryECLAT discovers the complete set of frequent patterns directly from a *binary* (0/1) transactional
                      dataset in which rows represent transactions and columns represent items. ECLAT is a vertical (tidset)
                      algorithm, so a one-hot / boolean matrix is already its native data layout and no flattening to an item-token
                      file is required. Every item column is packed into a bitset (``numpy.packbits``); itemsets are explored
                      depth-first over prefix-equivalence classes, and the support of an extension is the vectorised pop-count of the
                      bitwise-AND of the two parent bitsets.

    :**Reference**:  Mohammed Javeed Zaki: Scalable Algorithms for Association Mining. IEEE Trans. Knowl. Data Eng. 12(3):
                     372-390 (2000), https://ieeexplore.ieee.org/document/846291

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

      (.venv) $ python3 BinaryECLAT.py <inputFile> <outputFile> <minSup>

      Example Usage:

      (.venv) $ python3 BinaryECLAT.py binaryDB.txt patterns.txt 10.0

    .. note:: minSup can be specified  in support count or a value between 0 and 1.


    **Calling from a python program**

    .. code-block:: python

            import PAMI.frequentPattern.basic.BinaryECLAT as alg

            import numpy as np

            iFile = np.array([[1, 0, 1, 1], [0, 1, 1, 0]])  # rows = transactions, cols = items

            minSup = 1  # can also be specified between 0 and 1

            obj = alg.BinaryECLAT(iFile, minSup)

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
        self._packed = None          # uint8 bitsets, shape (nWords, nItems); column j is item j
        self._nTransactions = 0
        self._popcount = None

    def _peekHeader(self, path: str) -> bool:
        """
        Returns True if the first line of ``path`` contains non-numeric tokens, in which case it is
        treated as a header row carrying the item names.
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
        One-hot encode a list of item-token transactions into a boolean matrix and the sorted item vocabulary.
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
        Loads the binary dataset from the supported input forms, stores the item names in ``self._items`` and the
        bit-packed item columns in ``self._packed`` (column j is the bitset of item j).
        """
        matrix = None
        names = None

        if isinstance(self._iFile, _np.ndarray):
            matrix = self._iFile
            names = ["item" + str(j + 1) for j in range(matrix.shape[1])] if matrix.ndim == 2 else []

        elif isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                self._items, self._nTransactions = [], 0
                self._packed = _np.empty((0, 0), dtype=_np.uint8)
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
            self._items, self._nTransactions = [], 0
            self._packed = _np.empty((0, 0), dtype=_np.uint8)
            return

        boolMatrix = matrix.astype(bool)
        self._items = [str(n) for n in names]
        self._nTransactions = boolMatrix.shape[0]
        self._packed = _np.packbits(boolMatrix, axis=0)

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

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """
        Frequent pattern mining process will start from here
        """
        self.mine()

    def _recursive(self, indexTuples: List[Tuple[int, ...]], bitsets: _np.ndarray) -> None:
        """
        Depth-first ECLAT over a prefix-equivalence class. Every member shares the same prefix, so member ``i`` is
        extended with the last item of each member ``j > i``; the support of the extension is the pop-count of the
        AND of the two bitsets. The whole row ``j > i`` is intersected against member ``i`` in a single batched op.

        :param indexTuples: sorted integer-id tuples of the class members.
        :param bitsets: row-major uint8 bitsets, bitsets[i] is the AND of member i's columns.
        """
        m = len(indexTuples)
        for i in range(m):
            if i + 1 >= m:
                break
            # Intersect member i with every later member in one vectorised AND + pop-count.
            candBitsets = bitsets[i + 1:] & bitsets[i]
            supports = self._popcount[candBitsets].sum(axis=1)
            keep = _np.where(supports >= self._minSup)[0]
            if keep.size == 0:
                continue

            childTuples: List[Tuple[int, ...]] = []
            for pos in keep:
                j = i + 1 + int(pos)
                candidate = indexTuples[i] + (indexTuples[j][-1],)
                childTuples.append(candidate)
                self._finalPatterns[tuple(self._items[c] for c in candidate)] = int(supports[pos])

            if len(childTuples) > 1:
                self._recursive(childTuples, candBitsets[keep].copy())

    def mine(self) -> None:
        """
        Frequent pattern mining process will start from here.
        """
        self._startTime = _ab._time.time()
        self._finalPatterns = {}
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)

        if self._nTransactions == 0 or self._packed.shape[1] == 0:
            self._wrapUp()
            return

        self._popcount = _np.unpackbits(_np.arange(256, dtype=_np.uint8)[:, None], axis=1).sum(axis=1).astype(_np.int64)

        itemSupports = self._popcount[self._packed].sum(axis=0)
        frequentCols = _np.where(itemSupports >= self._minSup)[0]
        if frequentCols.size:
            # Sort ascending by support (mirrors ECLAT's processing order).
            order = _np.argsort(itemSupports[frequentCols], kind="stable")
            frequentCols = frequentCols[order]

        indexTuples: List[Tuple[int, ...]] = [(int(c),) for c in frequentCols]
        for c in frequentCols:
            self._finalPatterns[(self._items[c],)] = int(itemSupports[c])

        if frequentCols.size > 1:
            bitsets = self._packed[:, frequentCols].T.copy()
            self._recursive(indexTuples, bitsets)

        self._wrapUp()

    def _wrapUp(self) -> None:
        """Records end time and memory consumption."""
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using BinaryECLAT algorithm ")

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

    def getPatternsAsDataFrame(self) -> _ab._pd.DataFrame:
        """
        Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataFrame = _ab._pd.DataFrame(list([[self._sep.join(x), y] for x, y in self._finalPatterns.items()]),
                                      columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, oFile: str, seperator="\t") -> None:
        """
        Complete set of frequent patterns will be loaded in to an output file

        :param oFile: name of the output file
        :type oFile: csvfile
        :param seperator: variable to store separator value
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
        This function is used to print the result
        """
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = BinaryECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = BinaryECLAT(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.mine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
