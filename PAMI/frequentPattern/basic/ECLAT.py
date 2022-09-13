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


from PAMI.frequentPattern.basic import abstract as _ab


class ECLAT(_ab._frequentPatterns):
    """ ECLAT is one of the fundamental algorithm to discover frequent patterns in a transactional database.
        This program employs downward closure property to  reduce the search space effectively.
        This algorithm employs depth-first search technique to find the complete set of frequent patterns in a
        transactional database.

        Reference:
        ----------
            Mohammed Javeed Zaki: Scalable Algorithms for Association Mining. IEEE Trans. Knowl. Data Eng. 12(3):
            372-390 (2000), https://ieeexplore.ieee.org/document/846291

        Attributes:
        ----------
            iFile : str
                Input file name or path of the input file
            minSup: float or int or str
                The user can specify minSup either in count or proportion of database size.
                If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                Otherwise, it will be treated as float.
                Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
            sep : str
                This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
                However, the users can override their default separator.
            oFile : str
                Name of the output file or the path of the output file
            startTime:float
                To record the start time of the mining process
            endTime:float
                To record the completion time of the mining process
            finalPatterns: dict
                Storing the complete set of patterns in a dictionary variable
            memoryUSS : float
                To store the total amount of USS memory consumed by the program
            memoryRSS : float
                To store the total amount of RSS memory consumed by the program

        Methods:
        -------
            startMine()
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
            creatingItemSets()
                Scans the dataset or dataframes and stores in list format
            frequentOneItem()
                Generates one frequent patterns
            ECLATGeneration(candidateList)
                It will generate the combinations of frequent items
            generateFrequentPatterns(tidList)
                It will generate the combinations of frequent items from a list of items

        Executing the code on terminal:
        -------------------------------

            Format:
            ------
                python3 ECLAT.py <inputFile> <outputFile> <minSup>

            Examples:
            ---------
                python3 ECLAT.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in times of minSup and count of database transactions)

                python3 ECLAT.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)


        Sample run of the importing code:
        ---------------------------------

            import PAMI.frequentPattern.basic.ECLAT as alg

            obj = alg.ECLAT(iFile, minSup)

            obj.startMine()

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

        Credits:
        --------
            The complete program was written by Kundai  under the supervision of Professor Rage Uday Kiran.

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
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def _getUniqueItemList(self):
        """
        Generating one frequent patterns
        """
        self._finalPatterns = {}
        candidate = {}
        uniqueItem = []
        for i in range(len(self._Database)):
            for j in range(len(self._Database[i])):
                if self._Database[i][j] not in candidate:
                    candidate[self._Database[i][j]] = {i}
                else:
                    candidate[self._Database[i][j]].add(i)
        for key, value in candidate.items():
            supp = len(value)
            if supp >= self._minSup:
                self._finalPatterns[key] = [value]
                uniqueItem.append(key)
        uniqueItem.sort()
        return uniqueItem

    def _generateFrequentPatterns(self, candidateFrequent):
        """It will generate the combinations of frequent items

        :param candidateFrequent :it represents the items with their respective transaction identifiers

        :type candidateFrequent: list

        :return: returning transaction dictionary

        :rtype: dict
        """
        new_freqList = []
        for i in range(0, len(candidateFrequent)):
            item1 = candidateFrequent[i]
            i1_list = item1.split()
            for j in range(i + 1, len(candidateFrequent)):
                item2 = candidateFrequent[j]
                i2_list = item2.split()
                if i1_list[:-1] == i2_list[:-1]:
                    interSet = self._finalPatterns[item1][0].intersection(self._finalPatterns[item2][0])
                    if len(interSet) >= self._minSup:
                        newKey = item1 + "\t" + i2_list[-1]
                        self._finalPatterns[newKey] = [interSet]
                        new_freqList.append(newKey)
                else: break

        if len(new_freqList) > 0:
                self._generateFrequentPatterns(new_freqList)

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

    def startMine(self):
        """Frequent pattern mining process will start from here"""

        self._startTime = _ab._time.time()
        if self._iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self._minSup is None:
            raise Exception("Please enter the Minimum Support")
        self._creatingItemSets()
        self._minSup = self._convert(self._minSup)
        uniqueItemList = self._getUniqueItemList()
        self._generateFrequentPatterns(uniqueItemList)
        for x, y in self._finalPatterns.items():
            self._finalPatterns[x] = len(y[0])
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using ECLAT algorithm")

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
            data.append([a.replace('\t', ' '), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            patternsAndSupport = x.strip() + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:",  self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = ECLAT(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = ECLAT(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        print("Total number of Frequent Patterns:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print(_ap.getPatternsAsDataFrame())
        print("Total Memory in USS:",  _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
