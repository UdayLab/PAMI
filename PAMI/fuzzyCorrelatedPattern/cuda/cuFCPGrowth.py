# cuFCPGrowth is a fundamental algorithm to discover correlated fuzzy frequent patterns in a quantitative transactional database using CUDA. This program employs the downward closure property to reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of correlated fuzzy frequent patterns in a quantitative transactional database.
#
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------
#
#             import PAMI.fuzzyCorrelatedPattern.cuda.cuFCPGrowth as alg
#
#             obj = alg.cuFCPGrowth(iFile, minSup, minAllConf)
#
#             obj.mine()
#
#             correlatedFuzzyFrequentPatterns = obj.getPatterns()
#
#             print("Total number of Correlated Fuzzy Frequent Patterns:", len(correlatedFuzzyFrequentPatterns))
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
from PAMI.fuzzyCorrelatedPattern.cuda import abstract as _ab
# import abstract as _ab


class _Regions:
    """
    A helper class to fuzzify a raw quantity into its Low/Middle/High region membership values.

    :Attributes:

        low : float
            low region membership value
        middle : float
            middle region membership value
        high : float
            high region membership value
    """

    def __init__(self, quantity):
        self.low = 0.0
        self.middle = 0.0
        self.high = 0.0
        if 0 < quantity <= 1:
            self.low = 1.0
        elif 1 <= quantity < 6:
            self.low = float((-0.2 * quantity) + 1.2)
            self.middle = float((0.2 * quantity) - 0.2)
        elif 6 <= quantity <= 11:
            self.middle = float((-0.2 * quantity) + 2.2)
            self.high = float((0.2 * quantity) - 1.2)
        else:
            self.high = 1.0


class cuFCPGrowth(_ab._corelatedFuzzyFrequentPatterns):
    """
    :Description: cuFCPGrowth is a fundamental algorithm to discover correlated fuzzy frequent patterns using Cuda in a quantitative transactional database. This program employs the downward closure property to reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of correlated fuzzy frequent patterns in a quantitative transactional database.

    :Reference:   Lin, N.P., & Chueh, H. (2007). Fuzzy correlation rules mining.
                  https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.6053&rep=rep1&type=pdf

    :param  iFile: str :
                   Name of the Input file to mine complete set of correlated fuzzy frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of correlated fuzzy frequent patterns
    :param  minSup: int or float or str :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
    :param  minAllConf: float :
                   The user can specify minAllConf values within the range (0, 1).
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default separator is tab space. However, the users can override their default separator.

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

      (.venv) $ python3 cuFCPGrowth.py <inputFile> <outputFile> <minSup> <minAllConf>

      Example Usage:

      (.venv) $ python3 cuFCPGrowth.py sampleDB.txt patterns.txt 10.0 0.4

    .. note:: minSup will be considered in percentage of database transactions


    **Importing this algorithm into a python program**
    ----------------------------------------------------

    .. code-block:: python

             import PAMI.fuzzyCorrelatedPattern.cuda.cuFCPGrowth as alg

             obj = alg.cuFCPGrowth(iFile, minSup, minAllConf)

             obj.mine()

             correlatedFuzzyFrequentPatterns = obj.getPatterns()

             print("Total number of Correlated Fuzzy Frequent Patterns:", len(correlatedFuzzyFrequentPatterns))

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
    _minAllConf = float()
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

    def _creatingItemSets(self):
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._transactions, self._fuzzyValues = [], []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
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
                    items = parts[0].split(self._sep)
                    quantities = parts[1].split(self._sep)
                    self._transactions.append([x for x in items if x])
                    self._fuzzyValues.append([float(x) for x in quantities if x])
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.split("\n")[0]
                            parts = line.split(":")
                            parts[0] = parts[0].strip()
                            parts[1] = parts[1].strip()
                            items = parts[0].split(self._sep)
                            quantities = parts[1].split(self._sep)
                            self._transactions.append([x for x in items if x])
                            self._fuzzyValues.append([float(x) for x in quantities if x])
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value):
        """

        To convert the user specified minSup value

        :param value: user specified minSup value

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
        Fuzzifies every raw quantity into its Low/Middle/High region, keeps only the dominant
        (highest scalar cardinality) region per item, builds a dense fuzzy value vector per
        surviving (item, region), and returns those that satisfy minSup.

        The all-confidence ratio denominator for an (item, region) is the number of
        transactions in which that item has a non-zero membership value in that region
        (a transaction count, not a fuzzy-value sum) -- matching the scalar cardinality
        strategy used by the CPU FCPGrowth implementation.
        """
        regionSums = {}
        regionCounts = {}
        for i in range(self._dbLen):
            for item, quantity in zip(self._transactions[i], self._fuzzyValues[i]):
                regions = _Regions(quantity)
                values = {'L': regions.low, 'M': regions.middle, 'H': regions.high}
                if item not in regionSums:
                    regionSums[item] = {'L': 0.0, 'M': 0.0, 'H': 0.0}
                    regionCounts[item] = {'L': 0, 'M': 0, 'H': 0}
                for r, v in values.items():
                    if v > 0:
                        regionSums[item][r] += v
                        regionCounts[item][r] += 1

        # keep only the dominant region per item (ties favour L, then M, then H)
        dominantRegion = {}
        for item, sums in regionSums.items():
            dominantRegion[item] = max(('L', 'M', 'H'), key=lambda r: sums[r])

        ratioDenom = {}
        for item, region in dominantRegion.items():
            ratioDenom[((item, region),)] = regionCounts[item][region]

        ArraysAndItems = {}
        for i in range(self._dbLen):
            for item, quantity in zip(self._transactions[i], self._fuzzyValues[i]):
                region = dominantRegion[item]
                regions = _Regions(quantity)
                value = {'L': regions.low, 'M': regions.middle, 'H': regions.high}[region]
                key = ((item, region),)
                if key not in ArraysAndItems:
                    ArraysAndItems[key] = _ab._np.zeros(self._dbLen, dtype=_ab._np.float32)
                ArraysAndItems[key][i] = value

        self._ratioDenom = {}
        newArraysAndItems = {}
        for k, v in ArraysAndItems.items():
            support = float(v.sum())
            if support >= self._minSup:
                denom = ratioDenom[k]
                ratio = support / denom if denom > 0 else 0.0
                self._finalPatterns[k] = [support, ratio]
                self._ratioDenom[k] = denom
                newArraysAndItems[k] = _ab._cp.array(v)

        return newArraysAndItems

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self):
        """
        Correlated fuzzy frequent pattern mining process will start from here
        """
        self.mine()

    def mine(self):
        """
        Correlated fuzzy frequent pattern mining process will start from here
        """
        _ab._cp.cuda.Device(0).use()
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._dbLen = len(self._transactions)
        self._minSup = self._convert(self._minSup)
        self._minAllConf = float(self._minAllConf)

        ArraysAndItems = self.arraysAndItems()

        while len(ArraysAndItems) > 0:
            newArraysAndItems = {}
            keys = list(ArraysAndItems.keys())

            # every key is a sorted tuple of (item, region) pairs, so keys[i]/keys[j]
            # union cleanly regardless of level -- guard the union to grow by exactly
            # one (item, region) pair per step, matching a proper level-wise expansion
            pairsA, pairsB, unions = [], [], []
            seen = set()
            for i in range(len(ArraysAndItems)):
                iItems = {pair[0] for pair in keys[i]}
                for j in range(i + 1, len(ArraysAndItems)):
                    jItems = {pair[0] for pair in keys[j]}
                    if iItems & jItems:
                        continue
                    union = tuple(sorted(set(keys[i]) | set(keys[j])))
                    if len(union) != len(keys[i]) + 1 or union in self._finalPatterns or union in seen:
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
                    for row, idx in enumerate(survivors):
                        union = unions[idx]
                        keyA, keyB = keys[pairsA[idx]], keys[pairsB[idx]]
                        denom = max(self._ratioDenom[keyA], self._ratioDenom[keyB])
                        self._ratioDenom[union] = denom
                        support = float(supports[idx])
                        ratio = support / denom if denom > 0 else 0.0
                        if ratio >= self._minAllConf:
                            self._finalPatterns[union] = [support, ratio]
                        newArraysAndItems[union] = newMatrix[row]

            ArraysAndItems = newArraysAndItems

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Correlated fuzzy frequent patterns were generated successfully using cuFCPGrowth algorithm ")

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
        Storing final correlated fuzzy frequent patterns in a dataframe
        :return: returning correlated fuzzy frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append(["\t".join([".".join(pair) for pair in a]), b[0], b[1]])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support', 'Confidence'])
        return dataFrame

    def save(self, outFile):
        """
        Complete set of correlated fuzzy frequent patterns will be loaded in to an output file
        :param outFile: name of the output file
        :type outFile: csvfile
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = "\t".join([".".join(pair) for pair in x]) + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of correlated fuzzy frequent patterns after completion of the mining process
        :return: returning correlated fuzzy frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """
        This function is used to print results
        """
        print("Total number of Correlated Fuzzy Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in s:", self.getRuntime())

if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = cuFCPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = cuFCPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.mine()
        print("Total number of Correlated Fuzzy Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in s:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
