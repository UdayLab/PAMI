import sys
import validators
from urllib.request import urlopen
from PAMI.partialPeriodicPattern.closed.abstract import *


class PPPClose(partialPeriodicPatterns):
    """ PPPClose algorithm is used to discover the closed partial periodic patterns in temporal databases.
        It uses depth-first search.

        Reference:
        -------
        ...
        
        Attributes:
        ----------
            iFile : str
                Input file name or path of the input file
            oFile : str
                Name of the output file or path of the input file
            periodicSupport: int or float or str
                The user can specify periodicSupport either in count or proportion of database size.
                If the program detects the data type of periodicSupport is integer, then it treats periodicSupport is expressed in count.
                Otherwise, it will be treated as float.
                Example: periodicSupport=10 will be treated as integer, while periodicSupport=10.0 will be treated as float
            period: int or float or str
                The user can specify period either in count or proportion of database size.
                If the program detects the data type of period is integer, then it treats period is expressed in count.
                Otherwise, it will be treated as float.
                Example: period=10 will be treated as integer, while period=10.0 will be treated as float
            sep : str
                This variable is used to distinguish items from one another in a transaction. The default separator is tab space or \t.
                However, the users can override their default separator.
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

        Executing the code on terminal:
        -------
        Format:
        ------
            python3 PPPClose.py <inputFile> <outputFile> <periodicSupport> <period>

        Examples:
        --------
            python3 PPPClose.py sampleTDB.txt patterns.txt 0.3 0.4   (periodicSupport and period will be considered in percentage of database
        transactions)

            python3 PPPClose.py sampleTDB.txt patterns.txt 3 4     (periodicSupport and period will be considered in support count or frequency)


        Sample run of the imported code:
        --------------

            from PAMI.partialPeriodicPattern.closed import PPPClose as alg

            obj = alg.PPPClose("../basic/sampleTDB.txt", "2", "6")

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(periodicFrequentPatterns))

            obj.savePatterns("patterns")

            Df = obj.getPatternsAsDataFrame()

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

    periodicSupport = float()
    period = float()
    startTime = float()
    endTime = float()
    finalPatterns = {}
    Database = []
    iFile = " "
    oFile = " "
    sep = " "
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
    
    def creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        self.Database = []
        if isinstance(self.iFile, pd.DataFrame):
            timeStamp, data = [], []
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'TS' in i:
                timeStamp = self.iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self.iFile['Transactions'].tolist()
            for i in range(len(data)):
                tr = [timeStamp[i]]
                tr = tr + data[i]
                self.Database.append(tr)
            self.lno = len(self.Database)
        if isinstance(self.iFile, str):
            if validators.url(self.iFile):
                data = urlopen(self.iFile)
                for line in data:
                    self.lno += 1
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.Database.append(temp)
            else:
                try:
                    with open(self.iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            self.lno += 1
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            self.Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def OneLengthPartialItems(self):
        """
        To scan the database and extracts the 1-length periodic-frequent items
        Returns:
        -------
        Returns the 1-length periodic-frequent items
        """
        self.mapSupport = {}
        self.tidList = {}
        self.period = self.convert(self.period)
        for line in self.Database:
            n = int(line[0])
            for i in range(1, len(line)):
                si = line[i]
                if self.mapSupport.get(si) is None:
                    self.mapSupport[si] = [1, 0, n]
                    self.tidList[si] = [n]
                else:
                    self.mapSupport[si][0] += 1
                    period = abs(n - self.mapSupport[si][2])
                    if period <= self.period:
                        self.mapSupport[si][1] += 1
                    self.mapSupport[si][2] = n
                    self.tidList[si].append(n)
        for x, y in self.mapSupport.items():
            period = abs(self.lno - self.mapSupport[x][2])
            if period <= self.period:
                self.mapSupport[x][1] += 1
        self.periodicSupport = self.convert(self.periodicSupport)
        self.mapSupport = {k: v[1] for k, v in self.mapSupport.items() if v[1] >= self.periodicSupport}
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
        """
        To calculate the weight if pattern based on the respective timeStamps
        Parameters
        ----------
        tidSet: timeStamps of the pattern

        Returns
        -------
        the calculated weight of the timeStamps
        """
        hashcode = 0
        for i in tidSet:
            hashcode += i
        if hashcode < 0:
            hashcode = abs(0 - hashcode)
        return hashcode % self.tableSize

    def contains(self, itemSet, val, hashcode):
        """
        To check if the key(hashcode) is in dictionary(hashing) variable
        Parameters:
        ----------
            itemSet: generated periodic-frequent itemSet
            val: support and period of itemSet
            hashcode: the key generated in calculate() method for every itemSet

        Returns
        -------
            true if itemSet with same support present in dictionary(hashing) or else returns false
        """
        if self.hashing.get(hashcode) is None:
            return False
        for i in self.hashing[hashcode]:
            itemSetX = i
            if val == self.hashing[hashcode][itemSetX] and set(itemSetX).issuperset(itemSet):
                return True
        return False

    def getPeriodicSupport(self, timeStamps):
        """
        Calculates the period and support of timeStamps
        Parameters:
        ----------
            timeStamps: timeStamps of itemSet

        Returns:
        -------
            period and support
        """
        timeStamps.sort()
        sup = 0
        for j in range(len(timeStamps) - 1):
            per = abs(timeStamps[j + 1] - timeStamps[j])
            if per <= self.period:
                sup += 1
        return sup

    def save(self, prefix, suffix, tidSetX):
        """
        Saves the generated pattern which satisfies the closed property
        Parameters:
        ----------
            prefix: the prefix part of itemSet
            suffix: the suffix part of itemSet
            tidSetX: the timeStamps of the generated itemSet

        Returns:
        -------
            saves the closed periodic-frequent pattern

        """
        if prefix is None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        prefix = list(set(prefix))
        prefix.sort()
        val = self.getPeriodicSupport(tidSetX)
        if val >= self.periodicSupport:
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
        Parameters:
        ----------
            prefix: Prefix class of an itemSet
            itemSets: suffix items in periodicFrequentItems that satisfies the periodicSupport condition
            tidSets: timeStamps of items in itemSets respectively

        Returns:
        -------
            closed periodic patterns with length more than 2
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
            if len(y1) >= self.periodicSupport:
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
                if len(y) < self.periodicSupport:
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
        self.startTime = time.time()
        self.creatingItemSets()
        self.finalPatterns = {}
        periodicFrequentItems = self.OneLengthPartialItems()
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
                if len(y1) < self.periodicSupport:
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
        self.memoryUSS = float()
        self.memoryRSS = float()
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Closed periodic frequent patterns were generated successfully using PPPClose algorithm ")

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

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

            :return: returning frequent patterns in a dataframe

            :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support', 'Period'])
        return dataFrame

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

            :param outFile: name of the output file

            :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = x + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

            :return: returning frequent patterns

            :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = PPPClose(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = PPPClose(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of  Patterns:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

