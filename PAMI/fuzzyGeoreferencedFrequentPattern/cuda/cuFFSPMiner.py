# cuFFSPMiner is a fundamental algorithm to discover fuzzy geo-referenced (spatial) frequent patterns in a quantitative transactional database using CUDA. This program employs the downward closure property, restricted to spatial neighbours, to reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of fuzzy spatial frequent patterns in a quantitative transactional database.
#
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------
#
#             import PAMI.fuzzyGeoreferencedFrequentPattern.cuda.cuFFSPMiner as alg
#
#             obj = alg.cuFFSPMiner(iFile, nFile, minSup)
#
#             obj.mine()
#
#             fuzzySpatialFrequentPatterns = obj.getPatterns()
#
#             print("Total number of fuzzy spatial frequent patterns:", len(fuzzySpatialFrequentPatterns))
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
from PAMI.fuzzyGeoreferencedFrequentPattern.cuda import abstract as _ab
# import abstract as _ab


class cuFFSPMiner(_ab._fuzzySpatialFrequentPatterns):
    """
    :Description: cuFFSPMiner is a fundamental algorithm to discover fuzzy spatial frequent patterns using Cuda in a quantitative transactional database. This program employs the downward closure property, restricted to spatial neighbours, to reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of fuzzy spatial frequent patterns in a quantitative transactional database.

    :Reference:   P. Veena, B. S. Chithra, R. U. Kiran, S. Agarwal and K. Zettsu, "Discovering Fuzzy Frequent
                  Spatial Patterns in Large Quantitative Spatiotemporal databases," 2021 IEEE International Conference on Fuzzy Systems
                  (FUZZ-IEEE), 2021, pp. 1-8, doi: 10.1109/FUZZ45933.2021.9494594.

    :param  iFile: str :
                   Name of the Input file to mine complete set of fuzzy spatial frequent patterns
    :param  nFile: str :
                   Name of the neighbourhood file that contains the spatial neighbours of each item
    :param  oFile: str :
                   Name of the output file to store complete set of fuzzy spatial frequent patterns
    :param  minSup: int or float or str :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
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

      (.venv) $ python3 cuFFSPMiner.py <inputFile> <outputFile> <neighbourFile> <minSup>

      Example Usage:

      (.venv) $ python3 cuFFSPMiner.py sampleDB.txt patterns.txt neighbours.txt 10.0

    .. note:: minSup will be considered in percentage of database transactions


    **Importing this algorithm into a python program**
    ----------------------------------------------------

    .. code-block:: python

             import PAMI.fuzzyGeoreferencedFrequentPattern.cuda.cuFFSPMiner as alg

             obj = alg.cuFFSPMiner(iFile, nFile, minSup)

             obj.mine()

             fuzzySpatialFrequentPatterns = obj.getPatterns()

             print("Total number of fuzzy spatial frequent patterns:", len(fuzzySpatialFrequentPatterns))

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
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _nFile = " "
    _oFile = " "
    _sep = "\t"
    _memoryUSS = float()
    _memoryRSS = float()
    _transactions = []
    _fuzzyValues = []
    _mapItemNeighbours = {}
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
        Storing the complete transactions of the database/input file in a database variable.
        Items are read as pre-fuzzified `item.region` labels, each paired with its fuzzy membership value.
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

    def _mapNeighbours(self):
        """
        Storing the spatial neighbours of every base item from the neighbourhood file.
        """
        self._mapItemNeighbours = {}
        if isinstance(self._nFile, _ab._pd.DataFrame):
            data, items = [], []
            if self._nFile.empty:
                print("its empty..")
            i = self._nFile.columns.values.tolist()
            if 'items' in i:
                items = self._nFile['items'].tolist()
            if 'Neighbours' in i:
                data = self._nFile['Neighbours'].tolist()
            for k in range(len(items)):
                self._mapItemNeighbours[items[k]] = data[k]
        if isinstance(self._nFile, str):
            if _ab._validators.url(self._nFile):
                data = _ab._urlopen(self._nFile)
                for line in data:
                    line = line.decode("utf-8")
                    line = line.split("\n")[0]
                    parts = [i.rstrip() for i in line.split(self._sep)]
                    item = parts[0]
                    self._mapItemNeighbours[item] = parts[1:]
            else:
                try:
                    with open(self._nFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.split("\n")[0]
                            parts = [i.rstrip() for i in line.split(self._sep)]
                            item = parts[0]
                            self._mapItemNeighbours[item] = parts[1:]
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
        Builds the fuzzy value array of every `item.region` label, computes its support, and
        returns the arrays of the labels that satisfy minSup together with each label's own
        spatial neighbourhood (used to restrict candidate joins to spatially co-located items).
        """
        ArraysAndItems = {}

        for i in range(self._dbLen):
            for item, fuzzyValue in zip(self._transactions[i], self._fuzzyValues[i]):
                j = (item,)
                if j not in ArraysAndItems:
                    ArraysAndItems[j] = _ab._np.zeros(self._dbLen, dtype=_ab._np.float32)
                ArraysAndItems[j][i] = fuzzyValue

        self._commonNeighbours = {}
        newArraysAndItems = {}

        for k, v in ArraysAndItems.items():
            support = float(v.sum())
            if support >= self._minSup:
                self._finalPatterns[k] = support
                baseItem = k[0][0]
                self._commonNeighbours[k] = set(self._mapItemNeighbours.get(baseItem, []))
                newArraysAndItems[k] = _ab._cp.array(v)

        return newArraysAndItems

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for mining process. Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self):
        """
        Fuzzy spatial frequent pattern mining process will start from here
        """
        self.mine()

    def mine(self):
        """
        Fuzzy spatial frequent pattern mining process will start from here
        """
        _ab._cp.cuda.Device(0).use()
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._mapNeighbours()
        self._dbLen = len(self._transactions)
        self._minSup = self._convert(self._minSup)

        ArraysAndItems = self.arraysAndItems()

        while len(ArraysAndItems) > 0:
            newArraysAndItems = {}
            newCommonNeighbours = {}
            keys = list(ArraysAndItems.keys())

            # a candidate join is only attempted when keys[i] and keys[j] share all but one
            # label each (a proper Apriori-gen join), and the label unique to j is a common
            # spatial neighbour of every base point already in itemset i
            pairsA, pairsB, unions, newItemOf = [], [], [], {}
            seen = set()
            for i in range(len(ArraysAndItems)):
                setI = set(keys[i])
                neighboursOfI = self._commonNeighbours[keys[i]]
                for j in range(i + 1, len(ArraysAndItems)):
                    setJ = set(keys[j])
                    onlyInJ = setJ - setI
                    onlyInI = setI - setJ
                    if len(onlyInJ) != 1 or len(onlyInI) != 1:
                        continue
                    newItem = next(iter(onlyInJ))
                    if newItem[0] not in neighboursOfI:
                        continue
                    union = tuple(sorted(setI | setJ))
                    if union in self._finalPatterns or union in seen:
                        continue
                    seen.add(union)
                    pairsA.append(i)
                    pairsB.append(j)
                    unions.append(union)
                    newItemOf[union] = newItem

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
                        keyA = keys[pairsA[idx]]
                        newItem = newItemOf[union]
                        newCommonNeighbours[union] = self._commonNeighbours[keyA] & set(
                            self._mapItemNeighbours.get(newItem[0], []))
                        self._finalPatterns[union] = float(supports[idx])
                        newArraysAndItems[union] = newMatrix[row]

            ArraysAndItems = newArraysAndItems
            self._commonNeighbours = newCommonNeighbours

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Fuzzy spatial frequent patterns were generated successfully using cuFFSPMiner algorithm ")

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
        Storing final fuzzy spatial frequent patterns in a dataframe
        :return: returning fuzzy spatial frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append(["\t".join([x[0] for x in a]), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile):
        """
        Complete set of fuzzy spatial frequent patterns will be loaded in to an output file
        :param outFile: name of the output file
        :type outFile: csvfile
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = "\t".join([i[0] for i in x]) + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of fuzzy spatial frequent patterns after completion of the mining process
        :return: returning fuzzy spatial frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """
        This function is used to print results
        """
        print("Total number of Fuzzy Spatial Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in s:", self.getRuntime())

if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = cuFFSPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = cuFFSPMiner(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.mine()
        print("Total number of Fuzzy Spatial Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in s:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
