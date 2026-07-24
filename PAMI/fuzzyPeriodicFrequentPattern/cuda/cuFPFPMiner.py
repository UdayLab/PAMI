# cuFPFPMiner is a fundamental algorithm to discover fuzzy periodic frequent patterns in a quantitative temporal database using CUDA. This program employs the downward closure property to reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of fuzzy periodic frequent patterns in a quantitative temporal database.
#
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------
#
#             import PAMI.fuzzyPeriodicFrequentPattern.cuda.cuFPFPMiner as alg
#
#             obj = alg.cuFPFPMiner(iFile, minSup, maxPer)
#
#             obj.mine()
#
#             fuzzyPeriodicFrequentPatterns = obj.getPatterns()
#
#             print("Total number of Fuzzy Periodic Frequent Patterns:", len(fuzzyPeriodicFrequentPatterns))
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
"""

from deprecated import deprecated
from PAMI.fuzzyPeriodicFrequentPattern.cuda import abstract as _ab
# import abstract as _ab

class cuFPFPMiner(_ab._fuzzyPeriodicFrequentPatterns):
    """
    :Description: cuFPFPMiner is a fundamental algorithm to discover fuzzy periodic frequent patterns using Cuda in a quantitative temporal database. This program employs the downward closure property to reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of fuzzy periodic frequent patterns in a quantitative temporal database.

    :Reference:   R. U. Kiran et al., "Discovering Fuzzy Periodic-Frequent Patterns in Quantitative Temporal Databases,"
                  2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), Glasgow, UK, 2020, pp.
                  1-8, doi: 10.1109/FUZZ48607.2020.9177579.

    :param  iFile: str :
                   Name of the Input file to mine complete set of fuzzy periodic frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of fuzzy periodic frequent patterns
    :param  minSup: int or float or str :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
    :param  maxPer: int or float :
                   The user can specify maxPer either in count or proportion of database size. If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.

    :Attributes:

        startTime : float
          To record the start time of the mining process

        endTime : float
          To record the completion time of the mining process

        finalPatterns : dict
          Storing the complete set of patterns in a dictionary variable

        memoryUSS : float
          To store the total amount of USS memory consumed by the program

        memoryRSS : float
          To store the total amount of RSS memory consumed by the program

        Database : list
          To store the transactions of a database in list



    **Methods to execute code on terminal**
    ----------------------------------------------------

    .. code-block:: console

      Format:

      (.venv) $ python3 cuFPFPMiner.py <inputFile> <outputFile> <minSup> <maxPer>

      Example Usage:

      (.venv) $ python3 cuFPFPMiner.py sampleDB.txt patterns.txt 10.0 3

    .. note:: minSup will be considered in percentage of database transactions


    **Importing this algorithm into a python program**
    ----------------------------------------------------

    .. code-block:: python

             import PAMI.fuzzyPeriodicFrequentPattern.cuda.cuFPFPMiner as alg

             obj = alg.cuFPFPMiner(iFile, minSup, maxPer)

             obj.mine()

             fuzzyPeriodicFrequentPatterns = obj.getPatterns()

             print("Total number of Fuzzy Periodic Frequent Patterns:", len(fuzzyPeriodicFrequentPatterns))

             obj.save(oFile)

             Df = obj.getPatternsAsDataFrame()

             memUSS = obj.getMemoryUSS()

             print("Total Memory in USS:", memUSS)

             memRSS = obj.getMemoryRSS()

             print("Total Memory in RSS", memRSS)

             run = obj.getRuntime()

             print("Total ExecutionTime in seconds:", run)


    **Credits:**
    -------------

             The complete program was written by Mithun Thangaraj under the supervision of Professor Rage Uday Kiran.

    """

    _minSup = float()
    _maxPer = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = "\t"
    _memoryUSS = float()
    _memoryRSS = float()
    _transactions = []
    _fuzzyValues = []
    _ts = []
    _dbLen = 0

    _supportKernel = _ab._cp.RawKernel(r'''

    extern "C" __global__

    void supportKernel(const float *matrix, const unsigned int *pairsA, const unsigned int *pairsB,
                       float *supports, unsigned int numElements, unsigned int numPairs)
    {
        __shared__ float partial[256];
        unsigned int p = blockIdx.x;
        if (p >= numPairs) return;
        const float *a = matrix + (unsigned long long)pairsA[p] * numElements;
        const float *b = matrix + (unsigned long long)pairsB[p] * numElements;
        float s = 0.0f;
        for (unsigned int t = threadIdx.x; t < numElements; t += blockDim.x)
        {
            float va = a[t];
            float vb = b[t];
            s += va < vb ? va : vb;
        }
        partial[threadIdx.x] = s;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            supports[p] = partial[0];
    }

    ''', 'supportKernel')

    _periodKernel = _ab._cp.RawKernel(r'''

    extern "C" __global__

    void periodKernel(const float *matrix, const float *ts, float *maxPeriods,
                      unsigned int numElements, unsigned int numRows)
    {
        unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= numRows) return;
        const float *row = matrix + (unsigned long long)r * numElements;
        float prev = 0.0f;
        float maxGap = 0.0f;
        for (unsigned int t = 0; t < numElements; t++)
        {
            if (row[t] > 0.0f)
            {
                float gap = ts[t] - prev;
                if (gap > maxGap) maxGap = gap;
                prev = ts[t];
            }
        }
        maxPeriods[r] = maxGap;
    }

    ''', 'periodKernel')

    def _creatingItemSets(self):
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._transactions, self._fuzzyValues, self._ts = [], [], []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'TS' in i:
                self._ts = self._iFile['TS'].tolist()
            if 'Transactions' in i:
                self._transactions = self._iFile['Transactions'].tolist()
            if 'fuzzyValues' in i:
                self._fuzzyValues = self._iFile['fuzzyValues'].tolist()
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.decode("utf-8")
                    line = line.split("\n")[0]
                    parts = line.split(":")
                    parts[0] = parts[0].strip()
                    parts[1] = parts[1].strip()
                    items = [x for x in parts[0].split(self._sep) if x]
                    quantities = [float(x) for x in parts[1].split(self._sep) if x]
                    self._ts.append(int(items[0]))
                    self._transactions.append(items[1:])
                    self._fuzzyValues.append(quantities)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.split("\n")[0]
                            parts = line.split(":")
                            parts[0] = parts[0].strip()
                            parts[1] = parts[1].strip()
                            items = [x for x in parts[0].split(self._sep) if x]
                            quantities = [float(x) for x in parts[1].split(self._sep) if x]
                            self._ts.append(int(items[0]))
                            self._transactions.append(items[1:])
                            self._fuzzyValues.append(quantities)
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value):
        """

        To convert the user specified maxPer value

        :param value: user specified maxPer value

        :type value: int or float or str

        :return: converted type

        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self._dbLen * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self._dbLen * value)
            else:
                value = int(value)
        return value

    def arraysAndItems(self):
        """
        Builds the fuzzy value array of every item, computes the support and the maximum
        periodicity of the single items, and returns the arrays of the items that satisfy minSup.
        """
        ArraysAndItems = {}

        for i in range(self._dbLen):
            for item, fuzzyValue in zip(self._transactions[i], self._fuzzyValues[i]):
                j = tuple([item])
                if j not in ArraysAndItems:
                    ArraysAndItems[j] = _ab._np.zeros(self._dbLen, dtype=_ab._np.float32)
                ArraysAndItems[j][i] = fuzzyValue

        lastTs = self._ts[-1]
        newArraysAndItems = {}

        for k, v in ArraysAndItems.items():
            support = float(v.sum())
            if support >= self._minSup:
                maxPeriod = 0
                prev = None
                for idx in _ab._np.nonzero(v)[0]:
                    t = self._ts[idx]
                    if prev is not None and t != lastTs:
                        maxPeriod = max(maxPeriod, t - prev)
                    prev = t
                if maxPeriod <= self._maxPer:
                    self._finalPatterns[k] = [support, maxPeriod]
                newArraysAndItems[k] = _ab._cp.array(v)

        return newArraysAndItems

    def _maxPeriods(self, matrix):
        """
        Computes the maximum periodicity of every row of a fuzzy value matrix on the GPU.
        A row's periods are the gaps between the timestamps of its consecutive non-zero
        entries, including the gap from time 0 to the first occurrence.

        :param matrix: cupy matrix of fuzzy values, one candidate pattern per row
        :return: numpy array with the maximum period of each row
        """
        numRows = matrix.shape[0]
        tsArr = _ab._cp.asarray(self._ts, dtype=_ab._cp.float32)
        maxPeriods = _ab._cp.zeros(numRows, dtype=_ab._cp.float32)
        threads = 256
        blocks = (numRows + threads - 1) // threads
        self._periodKernel((blocks,), (threads,),
                           (matrix, tsArr, maxPeriods,
                            _ab._np.uint32(self._dbLen), _ab._np.uint32(numRows)))
        return maxPeriods.get()

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self):
        """
        Fuzzy periodic frequent pattern mining process will start from here
        """
        self.mine()

    def mine(self):
        """
        Fuzzy periodic frequent pattern mining process will start from here
        """
        _ab._cp.cuda.Device(0).use()
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._dbLen = len(self._transactions)
        self._minSup = float(self._minSup)
        self._maxPer = self._convert(self._maxPer)

        ArraysAndItems = self.arraysAndItems()

        while len(ArraysAndItems) > 0:
            newArraysAndItems = {}
            keys = list(ArraysAndItems.keys())

            pairsA, pairsB, unions = [], [], []
            seen = set()
            for i in range(len(ArraysAndItems)):
                iList = list(keys[i])
                for j in range(i + 1, len(ArraysAndItems)):
                    jList = list(keys[j])
                    union = tuple(sorted(set(iList + jList)))
                    if len(union) != len(iList) + 1 or union in seen:
                        continue
                    seen.add(union)
                    pairsA.append(i)
                    pairsB.append(j)
                    unions.append(union)

            if len(pairsA) > 0:
                numPairs = len(pairsA)
                matrix = _ab._cp.stack(list(ArraysAndItems.values()))
                pairsA = _ab._np.asarray(pairsA, dtype=_ab._np.uint32)
                pairsB = _ab._np.asarray(pairsB, dtype=_ab._np.uint32)
                supports = _ab._cp.zeros(numPairs, dtype=_ab._cp.float32)
                self._supportKernel((numPairs,), (256,),
                                    (matrix, _ab._cp.array(pairsA), _ab._cp.array(pairsB), supports,
                                     _ab._np.uint32(self._dbLen), _ab._np.uint32(numPairs)))
                supports = supports.get()

                survivors = _ab._np.where(supports >= self._minSup)[0]
                if len(survivors) > 0:
                    survA = _ab._cp.array(pairsA[survivors])
                    survB = _ab._cp.array(pairsB[survivors])
                    newMatrix = _ab._cp.minimum(matrix[survA], matrix[survB])
                    maxPeriods = self._maxPeriods(newMatrix)
                    for row, idx in enumerate(survivors):
                        union = unions[idx]
                        if maxPeriods[row] <= self._maxPer:
                            self._finalPatterns[union] = [float(supports[idx]), int(maxPeriods[row])]
                        newArraysAndItems[union] = newMatrix[row]

            ArraysAndItems = newArraysAndItems

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Fuzzy periodic frequent patterns were generated successfully using cuFPFPMiner algorithm ")

    def getMemoryUSS(self):
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function
        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function
        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """
        Calculating the total amount of runtime taken by the mining process
        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """
        Storing final fuzzy periodic frequent patterns in a dataframe
        :return: returning fuzzy periodic frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([" ".join(a), b[0], b[1]])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataFrame

    def save(self, outFile):
        """
        Complete set of fuzzy periodic frequent patterns will be loaded in to an output file
        :param outFile: name of the output file
        :type outFile: csvfile
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = "\t".join(x) + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of fuzzy periodic frequent patterns after completion of the mining process
        :return: returning fuzzy periodic frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """
        This function is used to print results
        """
        print("Total number of Fuzzy Periodic Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in s:", self.getRuntime())

if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = cuFPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = cuFPFPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.mine()
        print("Total number of Fuzzy Periodic Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in s:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
