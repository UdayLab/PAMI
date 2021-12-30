#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.
#  Copyright (C)  2021 Rage Uday Kiran

from PAMI.frequentPattern.basic import abstract as _ab


class ECLATbitset(_ab._frequentPatterns):
    """
    ECLATbitset is one of the fundamental algorithm to discover frequent patterns in a transactional database.
    This program implemented following the eclat bitset algorithm.

    Reference:
    ----------
        Zaki, M.J., Gouda, K.: Fast vertical mining using diffsets. Technical Report 01-1, Computer Science
            Dept., Rensselaer Polytechnic Institute (March 2001), https://doi.org/10.1145/956750.956788

    Attributes:
    -----------
        self.iFile : str
            Input file name or path of the input file
        minSup: float or int or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
            However, the users can override their default separator.
        self.oFile : str
            Name of the output file or path of the output file
        self.startTime:float
            To record the start time of the mining process
        self.endTime:float
            To record the completion time of the mining process
        self.finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        self.memoryUSS : float
            To store the total amount of USS memory consumed by the program
        self.memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        self.Database : list
            To store the complete set of transactions available in the input database/file

    Methods:
    -------
    startMine()
        Mining process will start from here
    getPatterns()
        Complete set of patterns will be retrieved with this function
    savePatterns(oFile)
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
    generationOfAllItems()
        It will generate the combinations of frequent items
    startMine()
        the main function to mine the patterns

    Executing the code on terminal:
    -------------------------------

        Format:
        -------
        python3 ECLATbitset.py <inputFile> <outputFile> <minSup>

        Examples:
        ---------
        python3 ECLATbitset.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

        python3 ECLATbitset.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)

    Sample run of the importing code:
    ---------------------------------

        import PAMI.frequentPattern.basic.ECLATbitset as alg

        obj = alg.ECLATbitset(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.savePatterns(oFile)

        Df = obj.getPatternInDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

    Credits:
    --------
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

        """
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _minSup = str()
    _memoryUSS = float()
    _memoryRSS = float()
    _Database = []
    _mapSupport = {}
    _lno = 0

    def _convert(self, value):
        """
        To convert the user specified minSup value

        :param value: user specified minSup value

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

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable

        """
        self._Database = []
        self._mapSupport = {}
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()

        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
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
        self._minSup = self._convert(self._minSup)

    def _OneFrequentItems(self):
        items = []
        p = {}
        for i in self._Database:
            for j in i:
                if j not in items:
                    items.append(j)
        for temp in self._Database:
            for j in items:
                count = 0
                if j in temp:
                    count = 1
                if j not in p:
                    p[j] = [count]
                else:
                    p[j].append(count)
        for x, y in p.items():
            if self._countSupport(y) >= self._minSup:
                self._mapSupport[x] = y
        pList = [key for key, value in sorted(self._mapSupport.items(), key=lambda x: (len(x[1])), reverse=True)]
        return pList

    @staticmethod
    def _countSupport(tids):
        """To count support of 1's in tids

        :param tids: bitset representation of itemSets

        :return:  count
        """
        count = 0
        for i in tids:
            if i == 1:
                count += 1
        return count

    def _save(self, prefix, suffix, tidSetX):
        """To save the patterns satisfying the minSup condition

        :param prefix: prefix item of itemSet

        :param suffix: suffix item of itemSet

        :param tidSetX: bitset representation of itemSet

        :return: saving the itemSet in to finalPatterns
        """
        if prefix is None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        count = self._countSupport(tidSetX)
        sample = str()
        for i in prefix:
            sample = sample + i + " "
        self._finalPatterns[sample] = count

    def _generationOfAll(self, prefix, itemSets, tidSets):
        """It will generate the combinations of frequent items with prefix and  list of items

            :param prefix: it represents the prefix item to form the combinations

            :type prefix: list

            :param itemSets: it represents the suffix items of prefix

            :type itemSets: list

            :param tidSets: represents the tidLists of itemSets

            :type tidSets: 2d list
        """
        if len(itemSets) == 1:
            i = itemSets[0]
            tidI = tidSets[0]
            self._save(prefix, [i], tidI)
            return
        for i in range(len(itemSets)):
            itemI = itemSets[i]
            if itemI is None:
                continue
            tidSetX = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetx = [itemI]
            for j in range(i + 1, len(itemSets)):
                itemJ = itemSets[j]
                tidSetJ = tidSets[j]
                y = [k & l for k, l in zip(tidSetX, tidSetJ)]
                support = self._countSupport(y)
                if support >= self._minSup:
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newprefix = list(set(itemSetx)) + prefix
            self._generationOfAll(newprefix, classItemSets, classTidSets)
            del classItemSets, classTidSets
            self._save(prefix, list(set(itemSetx)), tidSetX)
            # raise Exception("end of time")

    def startMine(self):
        """Frequent pattern mining process will start from here
        We start with the scanning the itemSets and store the bitsets respectively.
        We form the combinations of single items and  check with minSup condition to check the frequency of patterns
        """

        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        plist = self._OneFrequentItems()
        self._finalPatterns = {}
        for i in range(len(plist)):
            itemI = plist[i]
            tidSetX = self._mapSupport[itemI]
            itemSetx = [itemI]
            itemSets = []
            tidSets = []
            for j in range(i + 1, len(plist)):
                itemJ = plist[j]
                tidSetJ = self._mapSupport[itemJ]
                y1 = [k & l for k, l in zip(tidSetX, tidSetJ)]
                support = self._countSupport(y1)
                if support >= self._minSup:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self._generationOfAll(itemSetx, itemSets, tidSets)
            del itemSets, tidSets
            self._save(None, itemSetx, tidSetX)
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Eclat_bitset algorithm")

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process

        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self._finalPatterns


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = ECLATbitset(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = ECLATbitset(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        _Patterns = _ap.getPatterns()
        print("Total number of Frequent Patterns:", len(_Patterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        '''l = [2000]
        for i in l:
            ap = ECLATbitset('/Users/Likhitha/Downloads/mushrooms.txt', i, ' ')
            ap.startMine()
            Patterns = ap.getPatterns()
            print("Total number of Frequent Patterns:", len(Patterns))
            ap.savePatterns('/Users/Likhitha/Downloads/output')
            memUSS = ap.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = ap.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = ap.getRuntime()
            print("Total ExecutionTime in ms:", run)'''
        print("Error! The number of input parameters do not match the total number of parameters provided")