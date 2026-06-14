# FSPGrowth: given a transactional database and a spatial (or neighbourhood) file, FSPM aims to discover all of those
# patterns that satisfy the user-specified minimum support (minSup) and neighbourhood (maxDist) constraints.
#
# **Importing this algorithm into a python program**
# -------------------------------------------------------
#
#             from PAMI.georeferencedFrequentPattern.basic import FSPGrowth as alg
#
#             obj = alg.FSPGrowth("sampleTDB.txt", "sampleN.txt", 5)
#
#             obj.mine()
#
#             spatialFrequentPatterns = obj.getPatterns()
#
#             print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))
#
#             obj.save("outFile")
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

from PAMI.georeferencedFrequentPattern.basic import abstract as _ab
from typing import List, Dict, Union
from deprecated import deprecated


class _FPNode:
    """
    A compact node of the FP-tree used by the optimized mining engine.

    :Attributes:

        item : int or str
            The item stored at this node.
        parent : _FPNode
            Pointer to the parent node, used to walk prefix paths upward
            when building conditional pattern bases.
        count : int
            Support (weight) accumulated at this node.
        children : dict
            Mapping of child-item -> child node.
        node_link : _FPNode
            Link to the next node holding the same item, forming the
            header-table chain for this item.
    """

    __slots__ = ("item", "parent", "count", "children", "node_link")

    def __init__(self, item, parent):
        self.item = item
        self.parent = parent
        self.count = 0
        self.children = {}
        self.node_link = None


class FSPGrowth(_ab._spatialFrequentPatterns):
    """
    :Description:   Given a transactional database and a spatial (or neighbourhood) file, FSPM aims to discover all of
                    those patterns that satisfy the user-specified minimum support (minSup) and neighbourhood (maxDist)
                    constraints. A pattern is spatially valid when all of its items are mutual neighbours.

                    This implementation preserves the public API of the original PAMI FSPGrowth class but replaces the
                    earlier tree/prefix internals with a compact, parent-linked FP-tree, header-table mining, weighted
                    conditional pattern bases, dictionary-based frequency ordering, and recursive spatial pruning that
                    enforces the neighbourhood constraint during recursion rather than only at output time. It produces
                    the same set of patterns and supports as the original.

    :Reference:   Rage, Uday & Fournier Viger, Philippe & Zettsu, Koji & Toyoda, Masashi & Kitsuregawa, Masaru. (2020).
                  Discovering Frequent Spatial Patterns in Very Large Spatiotemporal Databases.

    :param  iFile: str or pd.DataFrame :
                   Name/path of the input transactional database, or a DataFrame containing a 'Transactions' column.
    :param  nFile: str or pd.DataFrame :
                   Name/path of the input neighbourhood file, or a DataFrame containing item and neighbour columns.
    :param  minSup: int or float or str :
                   The user can specify minSup either as a count or as a proportion of the database size. If the data
                   type of minSup is integer, it is treated as a count. Otherwise (float or decimal string), it is
                   treated as a proportion of the number of transactions.
    :param  sep: str :
                   Separator used to distinguish items within a transaction. Default is a tab space; users may override.

    :Attributes:

        iFile : str or pd.DataFrame
            Input file name/path of the transactional database.
        nFile : str or pd.DataFrame
            Input file name/path of the neighbourhood file.
        oFile : str
            Name/path of the output file.
        minSup : int or float
            Minimum support, as a count or as a proportion of database size.
        finalPatterns : dict
            Complete set of discovered patterns: tab-joined pattern string -> support.
        startTime : float
            Start time of the mining process.
        endTime : float
            Completion time of the mining process.
        memoryUSS : float
            Total USS memory consumed by the mining process.
        memoryRSS : float
            Total RSS memory consumed by the mining process.

    :Methods:

        mine()
            Starts the pattern-mining process.
        getPatterns()
            Returns the complete set of patterns as a dictionary.
        save(oFile)
            Writes the complete set of patterns to an output file.
        getPatternsAsDataFrame()
            Returns the complete set of patterns as a pandas DataFrame.
        getMemoryUSS()
            Returns the total USS memory consumed by the mining process.
        getMemoryRSS()
            Returns the total RSS memory consumed by the mining process.
        getRuntime()
            Returns the total runtime taken by the mining process.
        printResults()
            Prints a summary of the results.

    **Executing the code on terminal :**
    ----------------------------------------

    .. code-block:: console

      Format:

      (.venv) $ python3 FSPGrowth.py <inputFile> <outputFile> <neighbourFile> <minSup>

      Example Usage:

      (.venv) $ python3 FSPGrowth.py sampleTDB.txt output.txt sampleN.txt 0.5

    .. note:: A float minSup is interpreted as a proportion of the database transactions.


    **Sample run of importing the code :**
    ----------------------------------------
    .. code-block:: python

        from PAMI.georeferencedFrequentPattern.basic import FSPGrowth as alg

        obj = alg.FSPGrowth("sampleTDB.txt", "sampleN.txt", 5)

        obj.mine()

        spatialFrequentPatterns = obj.getPatterns()

        print("Total number of Spatial Frequent Patterns:", len(spatialFrequentPatterns))

        obj.save("outFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    **Credits:**
    --------------
        The original program was written by Yudai Masu under the supervision of Professor Rage Uday Kiran.
        This optimized, API-compatible implementation retains that public interface while replacing the internal
        mining engine.
    """

    _minSup = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _nFile = " "
    _oFile = " "
    _sep = "\t"
    _lno = 0
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _neighbourList = {}
    _fpList = []

    def __init__(self, iFile, nFile, minSup, sep="\t"):
        super().__init__(iFile, nFile, minSup, sep)
        self._iFile = iFile
        self._nFile = nFile
        self._minSup = minSup
        self._sep = sep
        self._Database = []
        self._neighbourList = {}
        self._NeighboursMap = {}
        self._fpList = []
        self._finalPatterns = {}
        self._lno = 0
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._startTime = float()
        self._endTime = float()

    def _split_line(self, line):
        """
        Split one input line using the configured separator, stripping empty tokens.

        :param line: A single raw line from the input.
        :type line: str
        :return: List of non-empty, right-stripped tokens.
        :rtype: list
        """
        return [x.rstrip() for x in line.strip().split(self._sep) if x.rstrip()]

    def _dedupe_transaction(self, transaction):
        """
        Remove duplicate items within a transaction while preserving order.

        :param transaction: A single transaction.
        :type transaction: list
        :return: Transaction with duplicate items removed.
        :rtype: list
        """
        return list(dict.fromkeys(transaction))

    def _creatingItemSets(self):
        """
        Read the transactional database into self._Database and set self._lno.
        Accepts a DataFrame, a file path, or a URL.
        """
        self._Database = []
        self._lno = 0

        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            columns = self._iFile.columns.values.tolist()
            if "Transactions" in columns:
                raw = self._iFile["Transactions"].tolist()
            elif "Patterns" in columns:
                raw = self._iFile["Patterns"].tolist()
            else:
                raw = []

            for transaction in raw:
                if isinstance(transaction, str):
                    transaction = self._split_line(transaction)
                elif not isinstance(transaction, list):
                    transaction = list(transaction)
                transaction = [str(x).rstrip() for x in transaction if str(x).rstrip()]
                if transaction:
                    self._Database.append(self._dedupe_transaction(transaction))

            self._lno = len(self._Database)
            return

        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.decode("utf-8")
                    transaction = self._split_line(line)
                    if transaction:
                        self._Database.append(self._dedupe_transaction(transaction))
            else:
                try:
                    with open(self._iFile, "r", encoding="utf-8") as f:
                        for line in f:
                            transaction = self._split_line(line)
                            if transaction:
                                self._Database.append(self._dedupe_transaction(transaction))
                except IOError:
                    print("File Not Found1")
                    quit()

        self._lno = len(self._Database)

    def _mapNeighbours(self):
        """
        Read the neighbourhood file. Maintains the original self._neighbourList (item -> list of neighbours) and an
        internal set-based self._NeighboursMap (item -> set of neighbours including the item itself) used for fast
        intersection during spatial pruning. Accepts a DataFrame, a file path, or a URL.
        """
        self._neighbourList = {}
        self._NeighboursMap = {}

        if isinstance(self._nFile, _ab._pd.DataFrame):
            if self._nFile.empty:
                print("its empty..")
            columns = self._nFile.columns.values.tolist()

            item_col = next((c for c in ("items", "item", "Items", "Item") if c in columns), None)
            neigh_col = next((c for c in ("Neighbours", "neighbors", "NeighboursMap", "neighbours") if c in columns), None)

            if item_col is not None and neigh_col is not None:
                items = self._nFile[item_col].tolist()
                neighbours = self._nFile[neigh_col].tolist()
                for item, neighs in zip(items, neighbours):
                    item = str(item).rstrip()
                    if isinstance(neighs, str):
                        neigh_set = set(self._split_line(neighs))
                    else:
                        neigh_set = set(str(x).rstrip() for x in neighs if str(x).rstrip())
                    self._neighbourList[item] = list(neigh_set)
                    neigh_set.add(item)
                    self._NeighboursMap[item] = neigh_set
            return

        if isinstance(self._nFile, str):
            if _ab._validators.url(self._nFile):
                data = _ab._urlopen(self._nFile)
                for line in data:
                    line = line.decode("utf-8")
                    parts = self._split_line(line)
                    if not parts:
                        continue
                    item = parts[0]
                    neigh_set = set(parts[1:])
                    self._neighbourList[item] = list(neigh_set)
                    neigh_set.add(item)
                    self._NeighboursMap[item] = neigh_set
            else:
                try:
                    with open(self._nFile, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = self._split_line(line)
                            if not parts:
                                continue
                            item = parts[0]
                            neigh_set = set(parts[1:])
                            self._neighbourList[item] = list(neigh_set)
                            neigh_set.add(item)
                            self._NeighboursMap[item] = neigh_set
                except IOError:
                    print("File Not Found2")
                    quit()

    def _readDatabase(self):
        """
        Backward-compatible wrapper for the original method name. The original FSPGrowth read both the transactional
        database and the neighbourhood file inside _readDatabase(); this preserves that behaviour for callers/tests
        that invoke the method directly.
        """
        self._creatingItemSets()
        self._mapNeighbours()

    def _convert(self, value):
        """
        Convert the user-specified minSup to an absolute count. Integers are treated as counts; floats and decimal
        strings are treated as proportions of the database size.

        :param value: User-specified minSup.
        :type value: int or float or str
        :return: Converted minimum support value.
        :rtype: int or float
        """
        if type(value) is int:
            return int(value)
        if type(value) is float:
            return self._lno * value
        if type(value) is str:
            if "." in value:
                return self._lno * float(value)
            return int(value)
        return value

    def _getFrequentItems(self):
        """
        Count 1-item supports and prepare self._fpList. Retained for compatibility with the original private method;
        the optimized mine() applies the same logic internally.
        """
        item_counts = _ab._defaultdict(int)
        for transaction in self._Database:
            for item in transaction:
                item_counts[item] += 1
        frequent = {item: cnt for item, cnt in item_counts.items() if cnt >= self._minSup}
        self._finalPatterns = dict(frequent)
        self._fpList = sorted(frequent.keys(), key=lambda i: (-frequent[i], i))
        return frequent

    def _sortTransaction(self):
        """
        Sort each transaction according to self._fpList. Retained for compatibility; uses a rank dictionary rather
        than repeated list.index() calls.
        """
        rank = {item: idx for idx, item in enumerate(self._fpList)}
        for idx, transaction in enumerate(self._Database):
            filtered = [item for item in transaction if item in rank]
            filtered.sort(key=lambda item: rank[item])
            self._Database[idx] = filtered

    def _createFPTree(self):
        """
        Compatibility placeholder for the original private method. The optimized implementation builds FP-trees inside
        _mine_fp_growth() because each conditional database has its own header table, so this method is intentionally
        not used by mine().
        """
        return None

    def _get_common_neighbours(self, itemset):
        """
        Return the common neighbour set of all items in itemset (the items that may extend it while keeping every pair
        mutual neighbours).

        :param itemset: Current pattern items.
        :type itemset: tuple or list
        :return: Set of items that are neighbours of every item in itemset.
        :rtype: set
        """
        if not itemset:
            return set()
        iterator = iter(itemset)
        first = next(iterator)
        common = set(self._NeighboursMap.get(first, set()))
        for item in iterator:
            common &= self._NeighboursMap.get(item, set())
            if not common:
                break
        return common

    # ----------------------------------------------------------------------
    # Optimized FP-Growth core with recursive spatial pruning
    # ----------------------------------------------------------------------

    def _mine_fp_growth(self, transactions, min_count, f_order, prefix=()):
        """
        Mine frequent spatial patterns from a weighted conditional database.

        :param transactions: List of (transaction, weight) pairs.
        :type transactions: list
        :param min_count: Absolute minimum support.
        :type min_count: int or float
        :param f_order: Global item -> rank mapping for canonical ordering.
        :type f_order: dict
        :param prefix: Current pattern prefix.
        :type prefix: tuple
        :return: Mapping of canonical pattern tuple -> support.
        :rtype: dict
        """
        item_counts = _ab._defaultdict(int)
        for transaction, weight in transactions:
            for item in transaction:
                item_counts[item] += weight

        frequent_items = {
            item: count
            for item, count in item_counts.items()
            if count >= min_count and item in f_order
        }
        if not frequent_items:
            return {}

        # Enforce the spatial constraint during recursion, not only at output time.
        if prefix:
            allowed = self._get_common_neighbours(prefix)
            frequent_items = {item: count for item, count in frequent_items.items() if item in allowed}
            if not frequent_items:
                return {}

        f_list = sorted(frequent_items.keys(), key=lambda item: f_order[item])
        valid_items = set(f_list)

        root = _FPNode(None, None)
        header_table = {item: [frequent_items[item], None, None] for item in f_list}

        for transaction, weight in transactions:
            filtered = sorted(
                (item for item in transaction if item in valid_items),
                key=lambda item: f_order[item],
            )
            if not filtered:
                continue
            node = root
            for item in filtered:
                child = node.children.get(item)
                if child is None:
                    child = _FPNode(item, node)
                    child.count = weight
                    node.children[item] = child
                    if header_table[item][1] is None:
                        header_table[item][1] = child
                        header_table[item][2] = child
                    else:
                        header_table[item][2].node_link = child
                        header_table[item][2] = child
                else:
                    child.count += weight
                node = child

        results = {}
        for item in reversed(f_list):
            new_pattern = prefix + (item,)
            canonical_pattern = tuple(sorted(new_pattern, key=lambda x: f_order[x]))
            results[canonical_pattern] = frequent_items[item]

            conditional_pattern_base = []
            current = header_table[item][1]
            while current is not None:
                path = []
                parent = current.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional_pattern_base.append((path[::-1], current.count))
                current = current.node_link

            if conditional_pattern_base:
                results.update(
                    self._mine_fp_growth(conditional_pattern_base, min_count, f_order, new_pattern)
                )

        return results

    def _pattern_to_key(self, pattern):
        """
        Convert a tuple/list pattern to the tab-joined string key used by getPatterns().

        :param pattern: Pattern as a tuple, list, or string.
        :return: Tab-joined string key.
        :rtype: str
        """
        if isinstance(pattern, str):
            return pattern
        if isinstance(pattern, (tuple, list)):
            return "\t".join(str(x) for x in pattern)
        return str(pattern)

    @deprecated("It is recommended to use 'mine()' instead of 'startMine()' for the mining process. "
                "Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self):
        """
        Start the pattern-mining process (deprecated alias for mine()).
        """
        self.mine()

    def mine(self):
        """
        Start the pattern-mining process.
        """
        self._startTime = _ab._time.time()
        self._finalPatterns = {}

        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")

        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._mapNeighbours()

        item_counts = _ab._defaultdict(int)
        for transaction in self._Database:
            for item in transaction:
                item_counts[item] += 1

        eligible = {item: count for item, count in item_counts.items() if count >= self._minSup}
        self._fpList = sorted(eligible.keys(), key=lambda item: (-eligible[item], item))
        f_order = {item: rank for rank, item in enumerate(self._fpList)}

        weighted_transactions = [(transaction, 1) for transaction in self._Database]
        mined = self._mine_fp_growth(weighted_transactions, self._minSup, f_order)

        self._finalPatterns = {
            self._pattern_to_key(pattern): support for pattern, support in mined.items()
        }

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent Spatial Patterns successfully generated using FSPGrowth")

    def getMemoryUSS(self):
        """
        Total amount of USS memory consumed by the mining process.

        :return: USS memory consumed.
        :rtype: float
        """
        return self._memoryUSS

    def getMemoryRSS(self):
        """
        Total amount of RSS memory consumed by the mining process.

        :return: RSS memory consumed.
        :rtype: float
        """
        return self._memoryRSS

    def getRuntime(self):
        """
        Total runtime taken by the mining process.

        :return: Runtime in seconds.
        :rtype: float
        """
        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """
        Store the discovered frequent spatial patterns in a DataFrame.

        :return: Patterns as a DataFrame with columns ['Patterns', 'Support'].
        :rtype: pd.DataFrame
        """
        data = []
        for pattern, support in self._finalPatterns.items():
            data.append([pattern.replace("\t", " "), support])
        return _ab._pd.DataFrame(data, columns=["Patterns", "Support"])

    def save(self, oFile):
        """
        Write the complete set of frequent patterns to an output file.

        :param oFile: Name of the output file.
        :type oFile: str
        """
        self._oFile = oFile
        with open(self._oFile, "w+", encoding="utf-8") as writer:
            for pattern, support in self._finalPatterns.items():
                writer.write("%s:%s \n" % (pattern.strip(), str(support)))

    def getPatterns(self):
        """
        Return the complete set of frequent patterns after mining.

        :return: Patterns as { tab-joined pattern string : support }.
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """
        Print a summary of the results.
        """
        print("Total number of Spatial Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())

class SpatialFPGrowth(FSPGrowth):
    pass


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = FSPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = FSPGrowth(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.mine()
        print("Total number of Spatial Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")