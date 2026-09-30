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

import math
from deprecated import deprecated
from PAMI.fuzzyFrequentPattern.basic import abstract as _ab


class _FuzzyList:
    def __init__(self, pattern, elements):
        self.pattern = pattern
        self.elements = elements
        self.support = math.fsum(element[1] for element in elements)
        self.resting = math.fsum(element[2] for element in elements)


class MFFIMiner(_ab._fuzzyFrequentPatterns):
    """
    Mine multiple fuzzy frequent itemsets using sparse fuzzy lists.

    All frequent regions are retained. A pattern contains at most one
    region of each base item. Support is the sum of transaction-wise
    minimum memberships. Depth-first search uses internal and resting
    fuzzy values to prune extensions.

    :Reference: Lin et al. Efficient Mining of Multiple Fuzzy Frequent
                Itemsets (2016). https://doi.org/10.1007/s40815-016-0246-1
    :param iFile: File, URL or DataFrame containing transformed fuzzy data.
                  Text rows use items:memberships, separated by sep.
                  Region labels use item.Region, for example milk.Low.
                  DataFrames use Transactions and fuzzyValues columns.
    :param minSup: Integer support count or float proportion of transactions.
    :param sep: Item separator, default tab.

    .. code-block:: python

        from PAMI.fuzzyFrequentPattern.basic import MFFIMiner as alg

        obj = alg.MFFIMiner('fuzzyDB.txt', 0.25)
        obj.mine()
        patterns = obj.getPatterns()
        obj.save('patterns.txt')

    .. code-block:: console

        python3 MFFIMiner.py fuzzyDB.txt patterns.txt 0.25
    """

    def __init__(self, iFile, minSup, sep='\t'):
        super().__init__(iFile, minSup, sep)
        if not isinstance(sep, str) or not sep:
            raise ValueError('sep must be a non-empty string')
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
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f'Row {row}: memberships must be finite values between 0 and 1')
        self._transactions.append(items)
        self._fuzzyValues.append(values)

    def _convert(self, value):
        if isinstance(value, str):
            value = float(value) if any(c in value.lower() for c in '.e') else int(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('minSup must be an integer count or float proportion')
        if not math.isfinite(value) or value <= 0:
            raise ValueError('minSup must be positive and finite')
        if isinstance(value, float):
            if value > 1:
                raise ValueError('A float minSup must be a proportion no greater than 1')
            return value * self._dbLen
        return value

    def _baseItem(self, label):
        return label.rsplit('.', 1)[0] if '.' in label else label

    def _buildFuzzyLists(self, minSup):
        values = {}
        for transaction, memberships in zip(self._transactions, self._fuzzyValues):
            for item, value in zip(transaction, memberships):
                values.setdefault(item, []).append(value)
        supports = {item: math.fsum(memberships) for item, memberships in values.items()}
        labels = sorted((item for item in supports if supports[item] >= minSup),
                        key=lambda item: (supports[item], item))
        rank = {item: index for index, item in enumerate(labels)}
        elements = {item: [] for item in labels}
        for tid, (transaction, memberships) in enumerate(zip(self._transactions, self._fuzzyValues)):
            entries = sorted(((item, value) for item, value in zip(transaction, memberships)
                              if item in rank and value > 0), key=lambda entry: rank[entry[0]])
            resting = 0.0
            for item, value in reversed(entries):
                elements[item].append((tid, value, resting))
                resting = max(resting, value)
        return [_FuzzyList((item,), elements[item]) for item in labels]

    def _construct(self, left, right):
        if self._baseItem(left.pattern[-1]) == self._baseItem(right.pattern[-1]):
            return None
        elements = []
        i = j = 0
        while i < len(left.elements) and j < len(right.elements):
            tidA, valueA, _ = left.elements[i]
            tidB, valueB, restingB = right.elements[j]
            if tidA < tidB:
                i += 1
            elif tidA > tidB:
                j += 1
            else:
                elements.append((tidA, min(valueA, valueB), restingB))
                i += 1
                j += 1
        return _FuzzyList(left.pattern + (right.pattern[-1],), elements)

    def _mineFuzzyLists(self, fuzzyLists, minSup):
        for index, current in enumerate(fuzzyLists):
            if current.support < minSup:
                continue
            self._finalPatterns[tuple(sorted(current.pattern))] = current.support
            if current.resting < minSup:
                continue
            extensions = []
            for nextIndex in range(index + 1, len(fuzzyLists)):
                joined = self._construct(current, fuzzyLists[nextIndex])
                if joined is not None and joined.support >= minSup:
                    extensions.append(joined)
            if extensions:
                self._mineFuzzyLists(extensions, minSup)

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()'.")
    def startMine(self):
        self.mine()

    def mine(self):
        self._startTime = _ab._time.time()
        self._finalPatterns = {}
        self._creatingItemSets()
        self._dbLen = len(self._transactions)
        minSup = self._convert(self._minSup)
        fuzzyLists = self._buildFuzzyLists(minSup)
        self._mineFuzzyLists(fuzzyLists, minSup)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print('Multiple fuzzy frequent patterns were generated successfully using MFFIMiner algorithm')

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
        obj = MFFIMiner(_ab._sys.argv[1], _ab._sys.argv[3], sep)
        obj.mine()
        obj.save(_ab._sys.argv[2])
        obj.printResults()
    else:
        print('Usage: python3 MFFIMiner.py <inputFile> <outputFile> <minSup> [sep]')
