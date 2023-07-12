# Association rule mining finds interesting associations and relationships among large sets of data items
#
# **Importing this algorithm into a python program**
# -----------------------------------------------------------
#
#
#           from PAMI.AssociationRules.basic import aprioriAlgorithm as alg
#           obj = alg.aprioriAlgorithm(iFile, frequentPatternsFile, threshold, sep)
#           obj.startMine()
#           Rules = obj.getPattern()
#           print("Total number of Patterns:", len(Patterns))
#           obj.savePatterns(oFile)
#           Df = obj.getPatternsAsDataFrame()
#           memUSS = obj.getMemoryUSS()
#           print("Total memory in USS", memUSS)
#           memRSS = obj.getMemoryRSS()
#           print("Total memory in RSS", memRSS)
#           run = obj.getRuntime()
#           print("Total ExecutionTime in seconds", run)


__copyright__ = """
Copyright (c) 2021 Rage Uday Kiran

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

from PAMI.AssociationRules.basic import abstract as _ab


class Confidence:
    """
    A class used to represent the confidence of frequentPattern


    Attributes :
    ----------------
        frequentPatterns : list
                storing the frequent patterns in a list
        singleItems : list
                storing the list of unique items
        threshold : int
                set an integer as threshold value
        finalPatterns : dic
                storing the obtained patterns in a dic

    Methods :
    --------------
        _generation(str1, str2)
            concat the items in a list by indexing and returns modified parameters
        _generaeWithConfidence(lhs, rhs)
            Calculate the min confidence of lhs and rhs and
            returns the finalFrequentPattern by comparing the
            confidence values with threshold value
        run()
            returns modified parameters by concat

"""

    def __init__(self, patterns, singleItems, threshold):
        """
        :param inputFile: input file name or path
        :type inputFile: str
        :param sep:
        """
        self._frequentPatterns = patterns
        self._singleItems = singleItems
        self._threshold = threshold
        self._finalPatterns = {}

    def _generation(self, prefix, suffix):
        if len(suffix) == 1:
            conf = self._generaeWithConfidence(prefix, suffix[0])
        for i in range(len(suffix)):
            suffix1 = suffix[:i] + suffix[i + 1:]
            prefix1 = prefix + ' ' + suffix[i]
            for j in range(i + 1, len(suffix)):
                self._generaeWithConfidence(prefix + ' ' + suffix[i], suffix[j])
                # self._generation(prefix+ ' ' +suffix[i], suffix[i+1:])
            self._generation(prefix1, suffix1)

    def _generaeWithConfidence(self, lhs, rhs):
        s = lhs + '\t' + rhs
        if self._frequentPatterns.get(s) == None:
            return 0
        minimum = self._frequentPatterns[s]
        conflhs = minimum / self._frequentPatterns[lhs]
        confrhs = minimum / self._frequentPatterns[rhs]
        if conflhs >= self._threshold:
            s1 = lhs + '->' + rhs
            self._finalPatterns[s1] = conflhs
        if confrhs >= self._threshold:
            s1 = rhs + '->' + lhs
            self._finalPatterns[s1] = confrhs

    def run(self):
        for i in range(len(self._singleItems)):
            suffix = self._singleItems[:i] + self._singleItems[i + 1:]
            prefix = self._singleItems[i]
            for j in range(i + 1, len(self._singleItems)):
                self._generaeWithConfidence(self._singleItems[i], self._singleItems[j])
            self._generation(prefix, suffix)


class ARWithConfidence:
    """
        temporalDatabaseStats is class to get stats of database.

        Attributes:
        --------------
        iFile : str
            stores path of the file
        threshold : int
            set a condition by integer value
        finalPatterns : {}
            stores the patterns in a dictionary
        sep :
            differentiate the items by using separator

        Methods:
        ------------
        _readPatterns()
            read the data in file and covert data into list
        startMine()
            run all methods
        getMemoryUSS()
            returns total USS memory consumed
        getMemoryRSS()
            returns the total RSS memory consumed
        getRuntime()
            returns the total time taken to complete the process
        getPatternsAsDataFrame()
            converting the data into dataframe
        save()
            saving the file
        getPatterns()
            returns the finalPatterns
        printResults()
            returns the total no of rules, memory, time consumed


    **Methods to execute code on terminal**
    ----------------------------------------
            Format:
                >>> python3 ARWithConfidence.py <inputFile> <outputFile>  <minSup> <sep>
            Example:
                >>>  python3 ARWithConfidence.py sampleTDB.txt output.txt sampleN.txt 0.25 0.2
                     .. note:: minSup will be considered in percentage of database transactions

    **Importing this algorithm into a python program**
    --------------------------------------------------------------------------------
    .. code-block:: python

            from PAMI.AssociationRules.basic import aprioriAlgorithm as alg
            obj = alg.aprioriAlgorithm(iFile, frequentPatternsFile, threshold, sep)
            obj.startMine()
            Rules = obj.getPatterns()
            print("Total number of  Patterns:", len(Patterns))
            obj.savePatterns(oFile)
            Df = obj.getPatternsAsDataFrame()
            memUSS = obj.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = obj.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = obj.getRuntime()
            print("Total ExecutionTime in seconds:", run)

    **Credits:**
    ----------------------------------------
            The complete program was written by ****** under the supervision of Professor Rage Uday Kiran.

    """

    def __init__(self, iFile, threshold, sep):
        """
        :param inputFile: input file name or path
        :type inputFile: str
        :param sep:
        """
        self._iFile = iFile
        self._threshold = threshold
        self._finalPatterns = {}
        self._sep = sep

    def _readPatterns(self):
        self._frequentPatterns = {}
        k = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            pattern, sup = [], []
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'pattern' in i:
                pattern = self._iFile['pattern'].tolist()
            if 'support' in i:
                support = self._iFile['support'].tolist()
            for i in range(len(pattern)):
                s = '\t'.join(pattern[i])
                self._frequentPattern[s] = support[i]
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line = line.strip()
                    line = line.split(':')
                    s = line[0].split(self._sep)
                    s = '\t'.join(s)
                    self._frequentPatterns[s.strip()] = int(line[1])
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            line = line.split(':')
                            s = line[0].split(self._sep)
                            for j in s:
                                if j not in k:
                                    k.append(j)
                            s = '\t'.join(s)
                            self._frequentPatterns[s.strip()] = int(line[1])
                except IOError:
                    print("File Not Found")
                    quit()
        return k

    def startMine(self):
        self._startTime = _ab._time.time()
        k = self._readPatterns()
        a = Confidence(self._frequentPatterns, k, self._threshold)
        a.run()
        self._finalPatterns = a._finalPatterns
        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Association rules successfully  generated from frequent patterns ")

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
        # dataFrame = dataFrame.replace(r'\r+|\n+|\t+',' ', regex=True)
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns

    def getPatterns(self):
        print("Total number of Association Rules:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = ARWithConfidence(_ab._sys.argv[1], _ab._sys.argv[3], float(_ab._sys.argv[4]))
        if len(_ab._sys.argv) == 4:
            _ap = ARWithConfidence(_ab._sys.argv[1], _ab._sys.argv[3])
        _ap.startMine()
        print("Total number of Association Rules:", len(_ap.getPatterns()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        _ap = ARWithConfidence('patterns.txt', 0.8, '\t')
        _ap.startMine()
        _ap.save('output.txt')
        _ap.printResults()
        print("Error! The number of input parameters do not match the total number of parameters provided")