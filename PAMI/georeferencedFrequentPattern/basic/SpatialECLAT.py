#  SpatialEclat is an Extension of ECLAT algorithm,which  stands for Equivalence Class Clustering and bottom-up
#  Lattice Traversal.It is one of the popular methods of Association Rule mining. It is a more efficient and
#  scalable version of the Apriori algorithm.
#
#  **Importing this algorithm into a python program**
#  ---------------------------------------------------
#
#             from PAMI.georeferencedFrequentPattern.basic import SpatialECLAT as alg
#
#             obj = alg.SpatialECLAT("sampleTDB.txt", "sampleN.txt", 5)
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
from deprecated import deprecated


class SpatialECLAT(_ab._spatialFrequentPatterns):
    """
    :Description:   Spatial Eclat is a Extension of ECLAT algorithm,which  stands for Equivalence Class Clustering and bottom-up
                    Lattice Traversal.It is one of the popular methods of Association Rule mining. It is a more efficient and
                    scalable version of the Apriori algorithm.

    :Reference:   Rage, Uday & Fournier Viger, Philippe & Zettsu, Koji & Toyoda, Masashi & Kitsuregawa, Masaru. (2020).
                  Discovering Frequent Spatial Patterns in Very Large Spatiotemporal Databases.

    :param  iFile: str :
                   Name of the Input file to mine complete set of Geo-referenced frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of Geo-referenced frequent patterns
    :param  minSup: int or float or str :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
    :param maxPer: float :
                   The user can specify maxPer in count or proportion of database size. If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
    :param nFile: str :
                   Name of the input file to mine complete set of Geo-referenced frequent patterns
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.


    :Attributes:

        iFile : str
            Input file name or path of the input file
        nFile : str
            Name of Neighbourhood file name
        minSup : int or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        startTime : float
            To record the start time of the mining process
        endTime : float
            To record the completion time of the mining process
        finalPatterns : dict
            Storing the complete set of patterns in a dictionary variable
        oFile : str
            Name of the output file to store complete set of frequent patterns
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        Database : list
            To store the complete set of transactions available in the input database/file

    :Methods:

        mine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingItemSets(iFileName)
            Storing the complete transactions of the database/input file in a database variable
        convert(value)
            To convert the given user specified value
        getNeighbourItems(keySet)
            A function to get common neighbours of a itemSet
        mapNeighbours(file)
            A function to map items to their neighbours

    **Executing the code on terminal :**
    ----------------------------------------

    .. code-block:: console

      Format:

      (.venv) $ python3 SpatialECLAT.py <inputFile> <outputFile> <neighbourFile> <minSup>

      Example Usage:

      (.venv) $ python3 SpatialECLAT.py sampleTDB.txt output.txt sampleN.txt 0.5

    .. note:: minSup will be considered in percentage of database transactions



    **Sample run of importing the code :**
    ------------------------------------------
    .. code-block:: python

        from PAMI.georeferencedFrequentPattern.basic import SpatialECLAT as alg

        obj = alg.SpatialECLAT("sampleTDB.txt", "sampleN.txt", 5)

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
    ----------------
        The complete program was written by B.Sai Chitra under the supervision of Professor Rage Uday Kiran.
    """

    _minSup = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _nFile = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _sep = "\t"

    def __init__(self, iFile, nFile, minSup, sep="\t"):
        super().__init__(iFile, nFile, minSup, sep)
        self._NeighboursMap = {}
        self._finalPatterns = {}

    def _creatingItemSets(self):
        """
        Storing the complete transactions of the database/input file in a database variable
        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
            if 'Patterns' in i:
                self._Database = self._iFile['Patterns'].tolist()

        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    if temp:
                        self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            if temp:
                                self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _convert(self, value):
        """
        To convert the given user specified value

        :param value: user specified value
        :type value: int or float or str
        :return: converted value
        :rtype: float
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

    def _mapNeighbours(self):
        """
        A function to map items to their Neighbours
        """
        self._NeighboursMap = {}
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
                item = items[k]
                neighbours = set(data[k])
                neighbours.add(item)
                self._NeighboursMap[item] = neighbours

        if isinstance(self._nFile, str):
            if _ab._validators.url(self._nFile):
                data = _ab._urlopen(self._nFile)
                for line in data:
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    if not temp:
                        continue
                    neighbours = set(temp[1:])
                    neighbours.add(temp[0])
                    self._NeighboursMap[temp[0]] = neighbours
            else:
                try:
                    with open(self._nFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = [i.rstrip() for i in line.strip().split(self._sep)]
                            parts = [x for x in parts if x]
                            if not parts:
                                continue
                            item = parts[0]
                            neighbours = set(parts[1:])
                            neighbours.add(item)
                            self._NeighboursMap[item] = neighbours
                except IOError:
                    print("File Not Found")
                    quit()

    def _getNeighbourItems(self, pattern):
        """
        A function to get common neighbours of a itemSet

        :param pattern: current pattern
        :type pattern: tuple
        :return: set of common neighbour items
        :rtype: set
        """
        if not pattern:
            return set()
        common = self._NeighboursMap.get(pattern[0], set())
        for i in range(1, len(pattern)):
            common = common.intersection(self._NeighboursMap.get(pattern[i], set()))
            if not common:
                break
        return common

    @deprecated(
        "It is recommended to use 'mine()' instead of 'startMine()' for the mining process. "
        "Starting from January 2025, 'startMine()' will be completely terminated.")
    def startMine(self):
        """
        Frequent pattern mining process will start from here
        """
        self.mine()

    def mine(self):
        """
        Frequent pattern mining process will start from here
        """
        self._startTime = _ab._time.time()
        self._finalPatterns = {}

        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")

        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        self._mapNeighbours()

        tid_list = {}
        for r_idx, trans in enumerate(self._Database):
            for item in trans:
                if item not in tid_list:
                    tid_list[item] = set()
                tid_list[item].add(r_idx)

        frequent_items = []
        for item in sorted(tid_list):
            tids = tid_list[item]
            if len(tids) >= self._minSup:
                self._finalPatterns[(item,)] = len(tids)
                frequent_items.append(((item,), tids))

        self._dfs(frequent_items)

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Spatial Frequent patterns were generated successfully using SpatialECLAT algorithm")

    def _dfs(self, siblings):
        """
        Recursive depth-first search.
        """
        for i in range(len(siblings)):
            pattern_a, tids_a = siblings[i]

            valid_neighbours = self._getNeighbourItems(pattern_a)

            children = []
            for j in range(i + 1, len(siblings)):
                pattern_b, tids_b = siblings[j]
                item_b = pattern_b[-1]

                if item_b not in valid_neighbours:
                    continue

                intersection = tids_a.intersection(tids_b)
                support = len(intersection)
                if support >= self._minSup:
                    new_pattern = pattern_a + (item_b,)
                    self._finalPatterns[new_pattern] = support
                    children.append((new_pattern, intersection))

            if children:
                self._dfs(children)

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
        data = []
        for a, b in self._finalPatterns.items():
            if isinstance(a, str):
                pat = a
            elif isinstance(a, (tuple, list)):
                pat = " ".join(str(x) for x in a)
            else:
                pat = str(a)
            data.append([pat.strip(), b])
        return _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])

    def save(self, outFile):
        """
        Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: str
        """
        self._oFile = outFile
        with open(self._oFile, 'w+', encoding='utf-8') as writer:
            for x, y in self._finalPatterns.items():
                if isinstance(x, str):
                    pat = x
                elif isinstance(x, (tuple, list)):
                    pat = "\t".join(str(i) for i in x)
                else:
                    pat = str(x)
                writer.write("%s:%s \n" % (pat.strip(), str(y)))

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        """
        This function is used to print the results
        """
        print("Total number of Spatial Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = SpatialECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = SpatialECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.mine()
        print("Total number of Spatial Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in seconds:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")