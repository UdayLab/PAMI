# cuAprioriBit is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs apriori property (or downward closure property) to  reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a transactional database.
#
#
# **Importing this algorithm into a python program**
# ----------------------------------------------------
#
#             import PAMI.frequentPattern.cuda.cuAprioriBit as alg
#
#             obj = alg.cuAprioriBit(iFile, minSup)
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

# from PAMI.frequentPattern.cuda import abstract as _ab
import abstract as _ab
from deprecated import deprecated


def _makeBatchedSupportKernel():
    """
    Build the batched support RawKernel used by cuAprioriBit.

    One thread computes the support (popcount of the bit-wise AND) of one candidate
    pair, so a whole candidate level is evaluated in a single kernel launch instead
    of one launch per pair.
    """
    return _ab._cp.RawKernel(r'''

    #define uint32_t unsigned int

    extern "C" __global__

    void batchedSupport(const uint32_t* M, const int* left, const int* right,
                        uint32_t* supports, int numPairs, int numWords)
    {
        int p = blockDim.x * blockIdx.x + threadIdx.x;
        if (p < numPairs)
        {
            const uint32_t* a = M + (size_t)left[p]  * numWords;
            const uint32_t* b = M + (size_t)right[p] * numWords;
            uint32_t s = 0;
            for (int w = 0; w < numWords; ++w)
            {
                s += __popc(a[w] & b[w]);
            }
            supports[p] = s;
        }
        return;
    }

    ''', 'batchedSupport')


class cuAprioriBit(_ab._frequentPatterns):
    """
    :Description: cuAprioriBit is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs apriori property (or downward closure property) to  reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a transactional database.

    :Reference:  Agrawal, R., Imieli ́nski, T., Swami, A.: Mining association rules between sets of items in large databases.
            In: SIGMOD. pp. 207–216 (1993), https://doi.org/10.1145/170035.170072

    :param  iFile: str :
                   Name of the Input file to mine complete set of frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of frequent patterns
    :param  minSup: int :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
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

      (.venv) $ python3 cuAprioriBit.py <inputFile> <outputFile> <minSup>

      Example Usage:

      (.venv) $ python3 cuAprioriBit.py sampleDB.txt patterns.txt 10.0

    .. note:: minSup will be considered in percentage of database transactions


    **Importing this algorithm into a python program**
    ----------------------------------------------------

    .. code-block:: python

            import PAMI.frequentPattern.cuda.cuAprioriBit as alg

            obj = alg.cuAprioriBit(iFile, minSup)

            obj.mine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.save(oFile)

            Df = obj.getPatternInDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


    **Credits:**
    -------------

             The complete program was written by Tarun Sreepada under the supervision of Professor Rage Uday Kiran.

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
    _Database = []

    _batchedSupportKernel = _makeBatchedSupportKernel()

    def _creatingItemSets(self):
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            temp = []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                temp = self._iFile['Transactions'].tolist()

            for k in temp:
                self._Database.append(set(k))
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(set(temp))
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(set(temp))
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
            value = (len(self._Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self._Database) * value)
            else:
                value = int(value)
        return value

    def arraysAndItems(self):
        ArraysAndItems = {}

        for i in range(len(self._Database)):
            for j in self._Database[i]:
                j = tuple([j])
                if j not in ArraysAndItems:
                    ArraysAndItems[j] = [i]
                else:
                    ArraysAndItems[j].append(i)

        newArraysAndItems = {}

        for k, v in ArraysAndItems.items():
            ArraysAndItems[k] = _ab._np.array(v, dtype=_ab._np.uint32)
            if len(v) >= self._minSup:
                self._finalPatterns[k] = len(v)
                newArraysAndItems[k] = ArraysAndItems[k]

        return newArraysAndItems

    def createBitRepresentation(self, ArraysAndItems):
        bitRep = {}
        arraySize = len(self._Database) // 32 + 1 if len(self._Database) % 32 != 0 else len(self._Database) // 32

        for k, v in ArraysAndItems.items():
            bitRep[k] = _ab._np.zeros(arraySize, dtype=_ab._np.uint32)
            for i in v:
                bitRep[k][i // 32] |= 1 << 31 - (i % 32)

        for k, v in bitRep.items():
            bitRep[k] = _ab._cp.array(v)

        return bitRep

    @deprecated("It is recommended to use 'mine()' instead of 'mine()' for mining process. Starting from January 2025, 'mine()' will be completely terminated.")
    def startMine(self):
        """
        Frequent pattern mining process will start from here
        """
        self.mine()

    def mine(self):
        """
        Frequent pattern mining process will start from here
        """
        self._Database = []
        self._startTime = _ab._time.time()
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)

        ArraysAndItems = self.arraysAndItems()
        ArraysAndItems = self.createBitRepresentation(ArraysAndItems)

        while len(ArraysAndItems) > 1:
            keys = list(ArraysAndItems.keys())
            numCands = len(keys)

            # Stack this level's bit-sets into one contiguous (numCands x numWords)
            # matrix so every candidate pair can be scored in a single kernel launch.
            M = _ab._cp.stack([ArraysAndItems[k] for k in keys])
            numWords = int(M.shape[1])

            # Enumerate all candidate pairs (i < j) once.
            iIdx, jIdx = _ab._np.triu_indices(numCands, k=1)
            numPairs = int(iIdx.shape[0])
            if numPairs == 0:
                break
            left = _ab._cp.asarray(iIdx, dtype=_ab._np.int32)
            right = _ab._cp.asarray(jIdx, dtype=_ab._np.int32)

            # One launch computes the support (popcount of the AND) of every pair.
            supports = _ab._cp.zeros(numPairs, dtype=_ab._np.uint32)
            threads = 256
            blocks = (numPairs + threads - 1) // threads
            self._batchedSupportKernel(
                (blocks,), (threads,),
                (M.ravel(), left, right, supports,
                 _ab._np.int32(numPairs), _ab._np.int32(numWords)))

            supportsHost = _ab._cp.asnumpy(supports)          # single device->host sync
            survivors = _ab._np.where(supportsHost >= self._minSup)[0]

            # Record the frequent unions (deduped within the level) and remember which
            # parent rows to AND together for the next level's bit-sets.
            newArraysAndItems = {}
            keepLeft, keepRight = [], []
            for p in survivors:
                i = int(iIdx[p])
                j = int(jIdx[p])
                union = tuple(sorted(set(keys[i]) | set(keys[j])))
                if union in newArraysAndItems:
                    continue
                newArraysAndItems[union] = None               # bit-set filled in below
                keepLeft.append(i)
                keepRight.append(j)
                self._finalPatterns["\t".join(union)] = int(supportsHost[p])

            if keepLeft:
                # Build every surviving candidate's bit-set in one vectorized GPU AND.
                kl = _ab._cp.asarray(keepLeft, dtype=_ab._np.int32)
                kr = _ab._cp.asarray(keepRight, dtype=_ab._np.int32)
                newM = _ab._cp.bitwise_and(M[kl], M[kr])
                for idx, union in enumerate(newArraysAndItems.keys()):
                    newArraysAndItems[union] = newM[idx]

            ArraysAndItems = newArraysAndItems

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using cuAprioriBit algorithm ")

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
        Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        # dataFrame = dataFrame.replace(r'\r+|\n+|\t+',' ', regex=True)
        return dataFrame

    def save(self, outFile):
        """
        Complete set of frequent patterns will be loaded in to an output file
        :param outFile: name of the output file
        :type outFile: csvfile
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
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
            _ap = cuAprioriBit(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = cuAprioriBit(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.mine()
        _ap.mine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")