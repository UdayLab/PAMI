import sys
from  abstract import *

class CPFPMiner(periodicFrequentPatterns):
    """ CPFPMiner algorithm is used to discover the closed periodic frequent patterns in temporal databases.
        It uses depth-first search.

        Reference : P. Likhitha et al., "Discovering Closed Periodic-Frequent Patterns in Very Large Temporal Databases"
        2020 IEEE International Conference on Big Data (Big Data), 2020, https://ieeexplore.ieee.org/document/9378215

        ...
        Attributes
        ----------

        iFile : str
            Input file name or path of the input file
        minSup: int/float/str
            UserSpecified minimum support value.
        maxPer :int/float/str
            UserSpecified minimum support value.
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        oFile : str
            Name of the output file to store complete set of frequent patterns
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program

        Methods
        -------

        startMine()
            Mining process will start from here
        getFrequentPatterns()
            Complete set of patterns will be retrieved with this function
        storePatternsInFile(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsInDataFrame()
            Complete set of frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function

        Executing the code on terminal:
        -------
        Format:
        ------
        python3 CPFPMiner.py <inputFile> <outputFile> <minSup> <maxPer>

        Examples:
        --------
        python3 CPFPMiner.py sampleTDB.txt patterns.txt 0.3 0.4   (minSup and maxPer will be considered in percentage of database
        transactions)

        python3 CPFPMiner.py sampleTDB.txt patterns.txt 3 4     (minSup and maxPer will be considered in support count or frequency)
        
        
        Sample run of the imported code:
        --------------
        from PAMI.periodicFrequentPattern.closed import cpfp as alg

        obj = alg.CPFPMiner("../basic/sampleTDB.txt", "2", "6")

        obj.startMine()

        periodicFrequentPatterns = obj.getPeriodicFrequentPatterns()

        print("Total number of Frequent Patterns:", len(periodicFrequentPatterns))

        obj.storePatternsInFile("patterns")

        Df = obj.getPatternsInDataFrame()

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

        Credits:
        -------
        The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.\n

        """

    minSup = float()
    maxPer = float()
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    memoryUSS = float()
    memoryRSS = float()
    transaction = []
    hashing = {}
    mapSupport = {}
    itemSetCount = 0
    maxItemId = 0
    tableSize = 10000
    tidList = {}
    lno = 0
    
    def convert(self, value):
        """
        To convert the given user specified value

        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self.lno * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self.lno * value)
            else:
                value = int(value)
        return value

    def convert(self, value):
        """
        To convert the given user specified value

        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self.lno * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self.lno * value)
            else:
                value = int(value)
        return value

    def scanDatabase(self):
        with open(self.iFile, 'r') as f:
            for line in f:
                self.lno += 1
                s = [i.rstrip() for i in line.split("\t")]
                n = int(s[0])
                for i in range(1, len(s)):
                    si = s[i]
                    if self.mapSupport.get(si) is None:
                        self.mapSupport[si] = [1, abs(0 - n), n]
                        self.tidList[si] = [n]
                    else:
                        self.mapSupport[si][0] += 1
                        self.mapSupport[si][1] = max(self.mapSupport[si][1], abs(n - self.mapSupport[si][2]))
                        self.mapSupport[si][2] = n
                        self.tidList[si].append(n)
        for x, y in self.mapSupport.items():
            self.mapSupport[x][1] = max(self.mapSupport[x][1], abs(self.lno - self.mapSupport[x][2]))
        self.minSup = self.convert(self.minSup)
        self.maxPer = self.convert(self.maxPer)
        self.mapSupport = {k: [v[0], v[1]] for k, v in self.mapSupport.items() if
                           v[0] >= self.minSup and v[1] <= self.maxPer}
        periodicFrequentItems = {}
        self.tidList = {k: v for k, v in self.tidList.items() if k in self.mapSupport}
        for x, y in self.tidList.items():
            t1 = 0
            for i in y:
                t1 += i
            periodicFrequentItems[x] = t1
        periodicFrequentItems = [key for key, value in sorted(periodicFrequentItems.items(), key=lambda x: x[1])]
        return periodicFrequentItems

    def calculate(self, tidSet):
        hashcode = 0
        for i in tidSet:
            hashcode += i
        if hashcode < 0:
            hashcode = abs(0 - hashcode)
        return hashcode % self.tableSize

    def contains(self, itemSet, val, hashcode):
        if self.hashing.get(hashcode) is None:
            return False
        for i in self.hashing[hashcode]:
            itemSetX = i
            if val[0] == self.hashing[hashcode][itemSetX][0] and set(itemSetX).issuperset(itemSet):
                return True
        return False

    def getPeriodAndSupport(self, timeStamps):
        """
        Calculates the support of periodicity of list of timeStamps
        :param timeStamps: timestamps of an itemSet
        :return: support and periodicity
        """
        timeStamps.sort()
        cur = 0
        per = 0
        sup = 0
        for j in range(len(timeStamps)):
            per = max(per, timeStamps[j] - cur)
            if per > self.maxPer:
                return [0, 0]
            cur = timeStamps[j]
            sup += 1
        per = max(per, self.lno - cur)
        return [sup, per]

    def save(self, prefix, suffix, tidSetX):
        """
        Saves the closed periodic frequent pattern
        :param prefix: prefix class of an itemSet
        :param suffix: suffix class of an itemSet
        :param tidSetX: timeStamps of an itemSet
        :return: saves closed periodic frequent pattern
        """
        if prefix is None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        prefix = list(set(prefix))
        prefix.sort()
        val = self.getPeriodAndSupport(tidSetX)
        if val[0] >= self.minSup and val[1] <= self.maxPer:
            hashcode = self.calculate(tidSetX)
            if self.contains(prefix, val, hashcode) is False:
                self.itemSetCount += 1
                sample = str()
                for i in prefix:
                    sample = sample + i + " "
                self.finalPatterns[sample] = val
            if hashcode not in self.hashing:
                self.hashing[hashcode] = {tuple(prefix): val}
            else:
                self.hashing[hashcode][tuple(prefix)] = val

    def processEquivalenceClass(self, prefix, itemSets, tidSets):
        """
        performs equivalence class to form the combination of items and results itemSets
        :param prefix: Prefix class of an itemSet
        :param itemSets: suffix items in periodicFrequentItems that satisfies the minSup condition
        :param tidSets: timeStamps of items in itemSets respectively
        :return: closed periodic patterns with length more than 2
        """
        if len(itemSets) == 1:
            i = itemSets[0]
            tidList = tidSets[0]
            self.save(prefix, [i], tidList)
            return
        if len(itemSets) == 2:
            itemI = itemSets[0]
            tidSetI = tidSets[0]
            itemJ = itemSets[1]
            tidSetJ = tidSets[1]
            y1 = list(set(tidSetI).intersection(tidSetJ))
            if len(y1) >= self.minSup:
                suffix = []
                suffix += [itemI, itemJ]
                suffix = list(set(suffix))
                self.save(prefix, suffix, y1)
            if len(y1) != len(tidSetI):
                self.save(prefix, [itemI], tidSetI)
            if len(y1) != len(tidSetJ):
                self.save(prefix, [itemJ], tidSetJ)
            return
        for i in range(len(itemSets)):
            itemX = itemSets[i]
            if itemX is None:
                continue
            tidSetX = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetX = [itemX]
            for j in range(i + 1, len(itemSets)):
                itemJ = itemSets[j]
                if itemJ is None:
                    continue
                tidSetJ = tidSets[j]
                y = list(set(tidSetX).intersection(tidSetJ))
                if len(y) < self.minSup:
                    continue
                if len(tidSetX) == len(tidSetJ) and len(y) == len(tidSetX):
                    itemSets.insert(j, None)
                    tidSets.insert(j, None)
                    itemSetX.append(itemJ)
                elif len(tidSetX) < len(tidSetJ) and len(y) == len(tidSetX):
                    itemSetX.append(itemJ)
                elif len(tidSetX) > len(tidSetJ) and len(y) == len(tidSetJ):
                    itemSets.insert(j, None)
                    tidSets.insert(j, None)
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
                else:
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            if len(classItemSets) > 0:
                newPrefix = list(set(itemSetX)) + prefix
                self.processEquivalenceClass(newPrefix, classItemSets, classTidSets)
            self.save(prefix, list(set(itemSetX)), tidSetX)

    def startMine(self):
        """
        Mining process will start from here
        """
        self.startTime = int(round(time.time() * 1000))
        periodicFrequentItems = self.scanDatabase()
        for i in range(len(periodicFrequentItems)):
            itemX = periodicFrequentItems[i]
            if itemX is None:
                continue
            tidSetX = self.tidList[itemX]
            itemSetX = [itemX]
            itemSets = []
            tidSets = []
            for j in range(i + 1, len(periodicFrequentItems)):
                itemJ = periodicFrequentItems[j]
                if itemJ is None:
                    continue
                tidSetJ = self.tidList[itemJ]
                y1 = list(set(tidSetX).intersection(tidSetJ))
                if len(y1) < self.minSup:
                    continue
                if len(tidSetX) == len(tidSetJ) and len(y1) is len(tidSetX):
                    periodicFrequentItems.insert(j, None)
                    itemSetX.append(itemJ)
                elif len(tidSetX) < len(tidSetJ) and len(y1) is len(tidSetX):
                    itemSetX.append(itemJ)
                elif len(tidSetX) > len(tidSetJ) and len(y1) is len(tidSetJ):
                    periodicFrequentItems.insert(j, None)
                    itemSets.append(itemJ)
                    tidSets.append(y1)
                else:

                    itemSets.append(itemJ)
                    tidSets.append(y1)
            if len(itemSets) > 0:
                self.processEquivalenceClass(itemSetX, itemSets, tidSets)
            self.save([], itemSetX, tidSetX)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Closed periodic frequent patterns were generated successfully using CPFPMiner algorithm ")

    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

            :return: returning USS memory consumed by the mining process
            :rtype: float
            """

        return self.memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

            :return: returning RSS memory consumed by the mining process
            :rtype: float
            """

        return self.memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process


            :return: returning total amount of runtime taken by the mining process
            :rtype: float
        """

        return self.endTime - self.startTime

    def getPatternsInDataFrame(self):
        """Storing final frequent patterns in a dataframe

            :return: returning frequent patterns in a dataframe
            :rtype: pd.DataFrame
            """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataFrame

    def storePatternsInFile(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

            :param outFile: name of the output file
            :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = x + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPeriodicFrequentPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

            :return: returning frequent patterns
            :rtype: dict
            """
        return self.finalPatterns
        

if __name__ == "__main__":
    if len(sys.argv) == 5:
        ap = CPFPMiner(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        frequentPatterns = ap.getPeriodicFrequentPatterns()
        print("Total number of Frequent Patterns:", len(frequentPatterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in seconds:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

