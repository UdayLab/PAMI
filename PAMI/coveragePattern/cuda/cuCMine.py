# cuCMine is a GPU accelerated algorithm to discover the coverage patterns in transactional databases.
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------
#
#
#             from PAMI.coveragePattern.cuda import cuCMine as alg
#
#             obj = alg.cuCMine(iFile, minRF, minCS, maxOR, seperator)
#
#             obj.mine()
#
#             coveragePattern = obj.getPatterns()
#
#             print("Total number of coverage Patterns:", len(coveragePattern))
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

from PAMI.coveragePattern.cuda import abstract as _ab
from typing import List, Dict, Tuple, Union
from deprecated import deprecated


class cuCMine(_ab._coveragePatterns):
    """
    About this algorithm
    ====================

    :Description:  cuCMine is a GPU accelerated version of CMine. Every coverage item is stored as a
                   bitset over the transactions of the database. The search is carried out level by
                   level instead of depth first, so that every extension of the current frontier is
                   evaluated in parallel on the GPU. Each candidate is handled by one thread block
                   that folds the bitwise AND of a prefix and an item into a population count.

                   The union cardinality is never counted directly. Because

                        |A union B| = |A| + |B| - |A intersection B|

                   the coverage support of an extension is obtained from the overlap count that the
                   pruning test already needs, so one population count per candidate is enough.

    :Reference:    Bhargav Sripada, Polepalli Krishna Reddy, Rage Uday Kiran:
                   Coverage patterns for efficient banner advertisement placement. WWW (Companion Volume) 2011: 131-132
                   __https://dl.acm.org/doi/10.1145/1963192.1963259

    :param  iFile: str :
                   Name of the Input file to mine complete set of coverage patterns
    :param  oFile: str :
                   Name of the output file to store complete set of coverage patterns
    :param  minRF: str:
                   Controls the minimum number of transactions in which every item must appear in a database.
    :param  minCS: str:
                   Controls the minimum number of transactions in which at least one time within a pattern must appear in a database.
    :param  maxOR: str:
                   Controls the maximum number of transactions in which any two items within a pattern can reappear.
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default separator is tab space.

    :Attributes:

        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime : float
            To record the start time of the mining process
        endTime : float
            To record the completion time of the mining process
        Database : list
            To store the transactions of a database in list

    Execution methods
    =================

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 cuCMine.py <inputFile> <outputFile> <minRF> <minCS> <maxOR> <separator>

      Example Usage:

      (.venv) $ python3 cuCMine.py sampleTDB.txt patterns.txt 0.4 0.7 0.5 ','

    **Calling from a python program**

    .. code-block:: python

            from PAMI.coveragePattern.cuda import cuCMine as alg

            obj = alg.cuCMine(iFile, minRF, minCS, maxOR, seperator)

            obj.mine()

            coveragePattern = obj.getPatterns()

            print("Total number of coverage Patterns:", len(coveragePattern))

            obj.save(oFile)

    Credits
    =======

             The complete program was written by Mithun Thangaraj under the supervision of Professor Rage Uday Kiran.
    """

    _startTime = float()
    _endTime = float()
    _minCS = str()
    _maxOR = str()
    _minRF = str()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _lno = 0
    _threadsPerBlock = 256
    _chunk = 1 << 20

    _kernels = _ab._cp.RawModule(code=r"""
// one block per candidate, fold the bitwise AND of the prefix bitset and the
// item bitset into a population count
extern "C" __global__ void overlapCounts(
    const unsigned long long* frontBits,
    const unsigned long long* itemBits,
    const int* candNode, const int* candItem,
    const long long nCands, const long long words,
    int* overlap)
{
    long long c = blockIdx.x;
    if (c >= nCands) return;

    const unsigned long long* a = frontBits + (long long) candNode[c] * words;
    const unsigned long long* b = itemBits + (long long) candItem[c] * words;

    unsigned int local = 0;
    for (long long w = threadIdx.x; w < words; w += blockDim.x)
        local += __popcll(a[w] & b[w]);

    __shared__ unsigned int buf[256];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) overlap[c] = (int) buf[0];
}

// one block per surviving candidate, write the union bitset of the next frontier
extern "C" __global__ void unionBits(
    const unsigned long long* frontBits,
    const unsigned long long* itemBits,
    const int* candNode, const int* candItem,
    const long long nSurv, const long long words,
    unsigned long long* outBits)
{
    long long s = blockIdx.x;
    if (s >= nSurv) return;

    const unsigned long long* a = frontBits + (long long) candNode[s] * words;
    const unsigned long long* b = itemBits + (long long) candItem[s] * words;
    unsigned long long* o = outBits + s * words;

    for (long long w = threadIdx.x; w < words; w += blockDim.x)
        o[w] = a[w] | b[w];
}
""", backend="nvrtc", options=("-std=c++14",))

    _overlapCounts = _kernels.get_function("overlapCounts")
    _unionBits = _kernels.get_function("unionBits")

    def _creatingItemSets(self) -> None:
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        self._lno = 0
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
            self._lno = len(self._Database)

        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
                    self._lno += 1
            else:
                try:
                    with open(self._iFile, 'r') as f:
                        for line in f:
                            self._lno += 1
                            splitter = [i.rstrip() for i in line.split(self._sep)]
                            splitter = [x for x in splitter if x]
                            self._Database.append(splitter)
                except IOError:
                    print("File Not Found")

    def creatingCoverageItems(self) -> Dict[str, List[int]]:
        """
        This function creates coverage items from _database.

        :return: coverageTidData that stores coverage items and their tid list.
        :rtype: dict
        """
        tidData = {}
        lno = 0
        for transaction in self._Database:
            for item in transaction:
                if item not in tidData:
                    tidData[item] = [lno]
                else:
                    tidData[item].append(lno)
            lno += 1
        self._lno = lno
        coverageTidData = {k: v for k, v in tidData.items() if len(v) / lno >= self._minRF}
        coverageTidData = dict(sorted(coverageTidData.items(), reverse=True, key=lambda x: len(x[1])))
        return coverageTidData

    def tidToBitset(self, itemSet: Dict[str, List[int]]) -> _ab._cp.ndarray:
        """
        This function converts the tid lists into one bitset matrix on the GPU. Every row is
        an item and every bit of a row is a transaction of the database.

        :param itemSet: coverage items and their tid lists
        :return: bitset matrix of shape (items, words)
        :rtype: cupy.ndarray
        """
        words = (self._lno + 63) // 64
        counts = _ab._np.array([len(v) for v in itemSet.values()], dtype=_ab._np.int64)
        allTids = _ab._np.concatenate([_ab._np.asarray(v, dtype=_ab._np.int64)
                                       for v in itemSet.values()])
        itemIdx = _ab._np.repeat(_ab._np.arange(counts.shape[0], dtype=_ab._np.int64), counts)

        flat = _ab._np.zeros(counts.shape[0] * words, dtype=_ab._np.uint64)
        position = itemIdx * words + (allTids >> 6)
        bit = _ab._np.uint64(1) << (allTids & 63).astype(_ab._np.uint64)
        _ab._np.bitwise_or.at(flat, position, bit)
        return _ab._cp.asarray(flat.reshape(counts.shape[0], words))

    def generateAllPatterns(self, itemNames: List[str], itemBits: _ab._cp.ndarray,
                            support: _ab._cp.ndarray) -> None:
        """
        This function generates all coverage patterns level by level. Every node of the current
        frontier carries the union bitset of the items it already holds, and is extended with
        every item that follows its last item. An extension survives when its overlap stays
        within maxOR, and it is reported when its coverage support reaches minCS.

        :param itemNames: names of the coverage items, ordered by descending support
        :param itemBits: bitset matrix of the coverage items
        :param support: number of transactions covered by every coverage item
        :return: None
        """
        itemCount = len(itemNames)
        words = itemBits.shape[1]
        minCoverage = self._minCS * self._lno

        frontBits = itemBits
        frontUnion = support
        prefix = _ab._np.arange(itemCount, dtype=_ab._np.int32).reshape(-1, 1)

        while prefix.shape[0] > 0:
            # every node is extended with the items that follow its last item
            last = prefix[:, -1].astype(_ab._np.int64)
            counts = itemCount - 1 - last
            total = int(counts.sum())
            if total == 0:
                break

            candNodeAll = _ab._np.repeat(_ab._np.arange(prefix.shape[0], dtype=_ab._np.int64), counts)
            offsets = _ab._np.repeat(_ab._np.cumsum(counts) - counts, counts)
            candItemAll = last[candNodeAll] + 1 + (_ab._np.arange(total, dtype=_ab._np.int64) - offsets)

            nextPrefix = []
            nextBits = []
            nextUnion = []

            for begin in range(0, total, self._chunk):
                end = min(begin + self._chunk, total)
                candNode = _ab._cp.asarray(candNodeAll[begin:end].astype(_ab._np.int32))
                candItem = _ab._cp.asarray(candItemAll[begin:end].astype(_ab._np.int32))
                nCands = int(candNode.shape[0])

                overlap = _ab._cp.zeros(nCands, dtype=_ab._cp.int32)
                self._overlapCounts(
                    (nCands,),
                    (self._threadsPerBlock,),
                    (frontBits, itemBits, candNode, candItem,
                     nCands, words, overlap)
                )

                # sorted closure property: only the overlap ratio limits the growth
                keep = overlap <= self._maxOR * support[candItem]
                survivor = _ab._cp.flatnonzero(keep)
                if survivor.shape[0] == 0:
                    continue

                candNode = candNode[survivor]
                candItem = candItem[survivor]
                nSurv = int(candNode.shape[0])

                # |A union B| = |A| + |B| - |A intersection B|
                union = frontUnion[candNode] + support[candItem] - overlap[survivor]

                outBits = _ab._cp.empty((nSurv, words), dtype=_ab._cp.uint64)
                self._unionBits(
                    (nSurv,),
                    (self._threadsPerBlock,),
                    (frontBits, itemBits, candNode, candItem,
                     nSurv, words, outBits)
                )

                nodeHost = _ab._cp.asnumpy(candNode)
                itemHost = _ab._cp.asnumpy(candItem)
                rows = _ab._np.concatenate(
                    [prefix[nodeHost], itemHost.reshape(-1, 1).astype(_ab._np.int32)], axis=1)

                reported = _ab._cp.asnumpy(_ab._cp.flatnonzero(union >= minCoverage))
                unionHost = _ab._cp.asnumpy(union)
                for r in reported:
                    name = '\t'.join(itemNames[c] for c in rows[r])
                    self._finalPatterns[name] = int(unionHost[r])

                nextPrefix.append(rows)
                nextBits.append(outBits)
                nextUnion.append(union)

            if not nextPrefix:
                break

            prefix = _ab._np.concatenate(nextPrefix)
            frontBits = _ab._cp.concatenate(nextBits)
            frontUnion = _ab._cp.concatenate(nextUnion)

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self) -> None:
        """ Main method to start """
        self.mine()

    def mine(self) -> None:
        """ Main method to start """

        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        self._creatingItemSets()
        self._minCS = self._convert(self._minCS)
        self._minRF = self._convert(self._minRF)
        self._maxOR = self._convert(self._maxOR)
        coverageItems = self.creatingCoverageItems()
        self._finalPatterns = {k: len(v) for k, v in coverageItems.items()}

        if coverageItems:
            itemNames = list(coverageItems.keys())
            support = _ab._cp.asarray(
                _ab._np.array([len(v) for v in coverageItems.values()], dtype=_ab._np.int64))
            itemBits = self.tidToBitset(coverageItems)
            self.generateAllPatterns(itemNames, itemBits, support)

        _ab._cp.cuda.Stream.null.synchronize()
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Coverage patterns were generated successfully using cuCMine algorithm")

    @staticmethod
    def _convert(value) -> Union[int, float]:
        """
        To convert the user specified value

        :param value: user specified value
        :return: converted type
        :rtype: Union[int, float]
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = value
        if type(value) is str:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        return value

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
        Storing final coverage patterns in a dataframe

        :return: returning coverage patterns in a dataframe
        :rtype: pd.DataFrame
        """
        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile: str) -> None:
        """
        Complete set of coverage patterns will be loaded in to an output file

        :param outFile: name of the output file
        :type outFile: file
        :return: None
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x.strip() + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self) -> Dict[str, int]:
        """
        Function to send the set of coverage patterns after completion of the mining process

        :return: returning coverage patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self) -> None:
        """
        This function is used to print the result
        """
        print("Total number of Coverage Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 7 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 7:
            _ap = cuCMine(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5], _ab._sys.argv[6])
        if len(_ab._sys.argv) == 6:
            _ap = cuCMine(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        _ap.mine()
        print("Total number of coverage Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
