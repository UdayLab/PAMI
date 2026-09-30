__copyright__ = """
Copyright (C) 2026 Rage Uday Kiran

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from itertools import groupby, islice
from deprecated import deprecated
from PAMI.fuzzyFrequentPattern.cuda import abstract as _ab


class cuMFFIMiner(_ab._fuzzyFrequentPatterns):
    """
    Mine multiple fuzzy frequent itemsets using CUDA.

    All frequent regions are retained. A pattern contains at most one
    region of each base item. Support is the sum of the minimum membership
    in each transaction. This implementation uses level-wise prefix joins
    and GPU reductions rather than the paper's depth-first fuzzy lists.

    :Reference: Lin et al. Efficient Mining of Multiple Fuzzy Frequent
                Itemsets (2016). https://doi.org/10.1007/s40815-016-0246-1
    :param iFile: File, URL or DataFrame containing transformed fuzzy data.
                  Text rows use items:memberships, separated by sep.
                  Region labels use item.Region, for example milk.Low.
                  DataFrames use Transactions and fuzzyValues columns.
    :param minSup: Integer support count or float proportion of transactions.
    :param sep: Item separator, default tab.
    :param batchSize: Maximum number of candidate pairs per kernel launch.

    .. code-block:: python

        from PAMI.fuzzyFrequentPattern.cuda import cuMFFIMiner as alg

        obj = alg.cuMFFIMiner('fuzzyDB.txt', 0.25)
        obj.mine()
        patterns = obj.getPatterns()
        obj.save('patterns.txt')

    .. code-block:: console

        python3 cuMFFIMiner.py fuzzyDB.txt patterns.txt 0.25
    """

    _supportKernel = _ab._cp.RawKernel(r'''
    extern "C" __global__
    void supportKernel(const double *matrix, const unsigned int *pairsA,
                       const unsigned int *pairsB, double *supports,
                       unsigned long long numElements)
    {
        __shared__ double partial[256];
        unsigned int p = blockIdx.x;
        const double *a = matrix + (unsigned long long)pairsA[p] * numElements;
        const double *b = matrix + (unsigned long long)pairsB[p] * numElements;
        double support = 0.0;
        for (unsigned long long t = threadIdx.x; t < numElements; t += blockDim.x)
            support += fmin(a[t], b[t]);
        partial[threadIdx.x] = support;
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

    _intersectionKernel = _ab._cp.RawKernel(r'''
    extern "C" __global__
    void intersectionKernel(const double *matrix, const unsigned int *pairsA,
                            const unsigned int *pairsB, double *output,
                            unsigned long long numElements)
    {
        unsigned int p = blockIdx.x;
        const double *a = matrix + (unsigned long long)pairsA[p] * numElements;
        const double *b = matrix + (unsigned long long)pairsB[p] * numElements;
        double *row = output + (unsigned long long)p * numElements;
        for (unsigned long long t = threadIdx.x; t < numElements; t += blockDim.x)
            row[t] = fmin(a[t], b[t]);
    }
    ''', 'intersectionKernel')

    def __init__(self, iFile, minSup, sep='\t', batchSize=4096):
        super().__init__(iFile, minSup, sep)
        if not isinstance(batchSize, int) or isinstance(batchSize, bool) or batchSize < 1:
            raise ValueError('batchSize must be a positive integer')
        if not isinstance(sep, str) or not sep:
            raise ValueError('sep must be a non-empty string')
        self._batchSize = batchSize
        self._transactions = []
        self._fuzzyValues = []
        self._dbLen = 0

    def _creatingItemSets(self):
        self._transactions = []
        self._fuzzyValues = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if not {'Transactions', 'fuzzyValues'}.issubset(self._iFile.columns):
                raise ValueError('DataFrame must contain Transactions and fuzzyValues')
            rows = zip(self._iFile['Transactions'], self._iFile['fuzzyValues'])
            for row, (items, values) in enumerate(rows, 1):
                if isinstance(items, str):
                    items = items.split(self._sep) if items.strip() else []
                if isinstance(values, str):
                    values = values.split(self._sep) if values.strip() else []
                self._addTransaction(items, values, row)
        else:
            source = (_ab._urlopen(self._iFile) if _ab._validators.url(self._iFile)
                      else open(self._iFile, encoding='utf-8'))
            with source:
                for row, line in enumerate(source, 1):
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    if not line.strip():
                        continue
                    parts = line.strip().split(':')
                    if len(parts) != 2:
                        raise ValueError(f'Row {row}: expected items:memberships')
                    items = parts[0].strip().split(self._sep) if parts[0].strip() else []
                    values = parts[1].strip().split(self._sep) if parts[1].strip() else []
                    self._addTransaction(items, values, row)

    def _addTransaction(self, items, values, row):
        items = list(items)
        values = list(values)
        if len(items) != len(values):
            raise ValueError(f'Row {row}: items and memberships must have the same length')
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f'Row {row}: item labels must be non-empty strings')
        items = [item.strip() for item in items]
        if len(set(items)) != len(items):
            raise ValueError(f'Row {row}: duplicate fuzzy item labels')
        values = [float(value) for value in values]
        if any(not _ab._math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f'Row {row}: memberships must be finite values between 0 and 1')
        self._transactions.append(items)
        self._fuzzyValues.append(values)

    def _convert(self, value):
        if isinstance(value, str):
            value = float(value) if any(c in value.lower() for c in '.e') else int(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('minSup must be an integer count or float proportion')
        if not _ab._math.isfinite(value) or value <= 0:
            raise ValueError('minSup must be positive and finite')
        if isinstance(value, float):
            if value > 1:
                raise ValueError('A float minSup must be a proportion no greater than 1')
            return value * self._dbLen
        return value

    def _baseItem(self, label):
        return label.rsplit('.', 1)[0] if '.' in label else label

    def _singletons(self, minSup):
        items = {}
        for tid, (transaction, values) in enumerate(zip(self._transactions, self._fuzzyValues)):
            for item, value in zip(transaction, values):
                if item not in items:
                    items[item] = _ab._np.zeros(self._dbLen, dtype=_ab._np.float64)
                items[item][tid] = value
        supports = {item: float(values.sum()) for item, values in items.items()}
        labels = sorted(item for item in items if supports[item] >= minSup)
        keys = [(item,) for item in labels]
        self._finalPatterns = {(item,): supports[item] for item in labels}
        if not keys:
            return keys, None
        matrix = _ab._cp.asarray(_ab._np.stack([items[item] for item in labels]))
        return keys, matrix

    def _candidatePairs(self, keys):
        frequent = set(keys)
        for prefix, group in groupby(enumerate(keys), key=lambda pair: pair[1][:-1]):
            group = list(group)
            used = {self._baseItem(item) for item in prefix}
            for pos, (i, left) in enumerate(group):
                leftBase = self._baseItem(left[-1])
                for nextPos in range(pos + 1, len(group)):
                    j, right = group[nextPos]
                    rightBase = self._baseItem(right[-1])
                    if rightBase == leftBase or rightBase in used:
                        continue
                    candidate = left + (right[-1],)
                    if all(candidate[:k] + candidate[k + 1:] in frequent
                           for k in range(len(prefix))):
                        yield i, j, candidate

    def _nextLevel(self, keys, matrix, minSup):
        candidates = self._candidatePairs(keys)
        newKeys = []
        matrices = []
        while True:
            batch = list(islice(candidates, self._batchSize))
            if not batch:
                break
            pairsA = _ab._np.asarray([pair[0] for pair in batch], dtype=_ab._np.uint32)
            pairsB = _ab._np.asarray([pair[1] for pair in batch], dtype=_ab._np.uint32)
            supports = _ab._cp.empty(len(batch), dtype=_ab._cp.float64)
            self._supportKernel((len(batch),), (256,),
                                (matrix, _ab._cp.asarray(pairsA), _ab._cp.asarray(pairsB),
                                 supports, _ab._np.uint64(self._dbLen)))
            supports = supports.get()
            survivors = _ab._np.flatnonzero(supports >= minSup)
            if not len(survivors):
                continue
            output = _ab._cp.empty((len(survivors), self._dbLen), dtype=_ab._cp.float64)
            self._intersectionKernel((len(survivors),), (256,),
                                     (matrix, _ab._cp.asarray(pairsA[survivors]),
                                      _ab._cp.asarray(pairsB[survivors]), output,
                                      _ab._np.uint64(self._dbLen)))
            matrices.append(output)
            for index in survivors:
                candidate = batch[index][2]
                newKeys.append(candidate)
                self._finalPatterns[candidate] = float(supports[index])
        if not newKeys:
            return newKeys, None
        return newKeys, matrices[0] if len(matrices) == 1 else _ab._cp.concatenate(matrices)

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()'.")
    def startMine(self):
        self.mine()

    def mine(self):
        self._startTime = _ab._time.time()
        self._finalPatterns = {}
        self._creatingItemSets()
        self._dbLen = len(self._transactions)
        minSup = self._convert(self._minSup)
        _ab._cp.cuda.Device(0).use()
        keys, matrix = self._singletons(minSup)
        while len(keys) > 1:
            keys, matrix = self._nextLevel(keys, matrix, minSup)
        _ab._cp.cuda.get_current_stream().synchronize()
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print('Multiple fuzzy frequent patterns were generated successfully using cuMFFIMiner algorithm')

    def getPatterns(self):
        return self._finalPatterns

    def getPatternsAsDataFrame(self):
        return _ab._pd.DataFrame([(self._sep.join(pattern), support)
                                 for pattern, support in self._finalPatterns.items()],
                                columns=['Patterns', 'Support'])

    def save(self, outFile):
        self._oFile = outFile
        with open(outFile, 'w', encoding='utf-8') as writer:
            for pattern, support in self._finalPatterns.items():
                writer.write(f'{self._sep.join(pattern)}:{support}\n')

    def getMemoryUSS(self):
        return self._memoryUSS

    def getMemoryRSS(self):
        return self._memoryRSS

    def getRuntime(self):
        return self._endTime - self._startTime

    def printResults(self):
        print('Total number of Multiple Fuzzy Frequent Patterns:', len(self.getPatterns()))
        print('Total Memory in USS:', self.getMemoryUSS())
        print('Total Memory in RSS:', self.getMemoryRSS())
        print('Total ExecutionTime in s:', self.getRuntime())


if __name__ == '__main__':
    if len(_ab._sys.argv) in (4, 5):
        sep = _ab._sys.argv[4] if len(_ab._sys.argv) == 5 else '\t'
        obj = cuMFFIMiner(_ab._sys.argv[1], _ab._sys.argv[3], sep)
        obj.mine()
        obj.save(_ab._sys.argv[2])
        obj.printResults()
    else:
        print('Usage: python3 cuMFFIMiner.py <inputFile> <outputFile> <minSup> [sep]')
