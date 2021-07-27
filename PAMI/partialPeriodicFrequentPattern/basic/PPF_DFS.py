import sys
from PAMI.partialPeriodicFrequentPattern.basic.abstract import *


class PPF_DFS(partialPeriodicPatterns):
    """
    PPF_DFS is algorithm to mine the partial periodic frequent patterns.

    Attributes
    ----------

    path : file
        input file path
    output : file
        output file name
    minSup : float
        user defined minsup
    maxPer : float
        user defined maxPer
    minPR : float
        user defined minPR
    tidlist : dict
        it store tids each item
    last : int
        it represents last time stamp in database
    lno : int
        number of line in database
    mapSupport : dict
        to maintain the information of item and their frequency
    finalPatterns : dict
        it represents to store the patterns
    runTime : float
        storing the total runtime of the mining process
    memoryUSS : float
        storing the total amount of USS memory consumed by the program
    memoryRSS : float
        storing the total amount of RSS memory consumed by the program

    Methods
    -------
    findDelimiter(line)
        Identifying the delimiter of the input file
    getPer_Sup(tids)
        caluclate ip / (sup+1)
    getPerSup(tids)
        caluclate ip
    scanDatabase(path)
        scan all lines in database
    sava(prefix,suffix,tidsetx)
        save prefix pattern with support and periodic ratio
    Generation(prefix, itemsets, tidsets)
        Userd to implement prefix class equibalence method to generate the periodic patterns recursively
    startMine()
        Mining process will start from here
    getPartialPeriodicPatterns()
        Complete set of patterns will be retrieved with this function
    storePatterns InFile(ouputFile)
        Complete set of frequent patterns will be loaded in to a ouput file
    getPatternsInDataFrame()
        Complete set of frequent patterns will be loaded in to a ouput file
    getMemoryUSS()
        Total amount of USS memory consumed by the mining process will be retrieved from this function
    getMemoryRSS()
        Total amount of RSS memory consumed by the mining process will be retrieved from this function
    getRuntime()
        Total amount of runtime taken by the mining process will be retrieved from this function

    Format: python3 PPF_DFS.py <inputFile> <outputFile> <minSup> <maxPer> <minPR>
    Examples: python3 PPF_DFS.py sampleDB.txt patterns.txt 10 10 0.5
    """
    path = ' '
    iFile = ' '
    oFile = ' '
    minSup = float()
    maxPer = float()
    minPR = float()
    tidlist = {}
    last = 0
    lno = 0
    mapSupport = {}
    finalPatterns = {}
    runTime = float()
    memoryUSS = float()
    memoryRSS = float()
    startTime = float()
    endTime = float()

    def findDelimiter(self, line):
        """Identifying the delimiter of the input file
            :param line: list of special characters may be used by a user to split the items in a input file
            :type line: list of string
            :returns: Delimited string used in the input file to split each item
            :rtype: string
            """
        l = ['\t', ' ', '*', '&', ' ', '%', '$', '#', '@', '!', '    ', '*', '(', ')']
        j = None
        for i in l:
            if i in line:
                return i
        return j

    def getPer_Sup(self, tids):
        """
        calculate ip / (sup+1)
        :param tids: it represent tid list
        :type tids: list
        :return: ip / (sup+1)
        """
        # print(lno)
        tids = list(set(tids))
        tids.sort()
        per = 0
        sup = 0
        cur = 0
        if len(tids) == 0:
            return 0
        if abs(0 - tids[0]) <= self.maxPer:
            sup += 1
        for j in range(len(tids) - 1):
            i = j + 1
            per = abs(tids[i] - tids[j])
            if (per <= self.maxPer):
                sup += 1
            cur = tids[j]
        if abs(self.last - tids[len(tids) - 1]) <= self.maxPer:
            sup += 1
        if sup == 0:
            return 0
        return sup / (len(tids) + 1)

    def getPerSup(self, tids):
        """
        calculate ip of a pattern
        :param tids: tid list of the pattern
        :type tids: list
        :return: ip
        """
        # print(lno)
        tids = list(set(tids))
        tids.sort()
        per = 0
        sup = 0
        cur = 0
        if len(tids) == 0:
            return 0
        if abs(0 - tids[0]) <= self.maxPer:
            sup += 1
        for j in range(len(tids) - 1):
            i = j + 1
            per = abs(tids[i] - tids[j])
            if (per <= self.maxPer):
                sup += 1
        if abs(tids[len(tids) - 1] - self.last) <= self.maxPer:
            sup += 1
        if sup == 0:
            return 0
        return sup

    def scanDatabase(self, path):
        """
        scan all lines of database and create support list
        :param path: it represents input file name
        :return: support list each item
        """
        id1 = 0
        with open(path, 'r') as f:
            for line in f:
                self.lno += 1
                # first=min(first,lno)
                # last=max(last,lno)
                line = line.strip()
                delimiter = self.findDelimiter([*line])
                s = [i.rstrip() for i in line.split(delimiter)]
                n = int(s[0])
                self.last = max(self.last, n)
                for i in range(1, len(s)):
                    si = s[i]
                    if abs(0 - n) <= self.maxPer:
                        if si not in self.mapSupport:
                            self.mapSupport[si] = [1, 1, n]
                            self.tidlist[si] = [n]
                        else:
                            lp = abs(n - self.mapSupport[si][2])
                            if lp <= self.maxPer:
                                self.mapSupport[si][0] += 1
                            self.mapSupport[si][1] += 1
                            self.mapSupport[si][2] = n
                            self.tidlist[si].append(n)
                    else:
                        if si not in self.mapSupport:
                            self.mapSupport[si] = [0, 1, n]
                            self.tidlist[si] = [n]
                        else:
                            lp = abs(n - self.mapSupport[si][2])
                            if lp <= self.maxPer:
                                self.mapSupport[si][0] += 1
                            self.mapSupport[si][1] += 1
                            self.mapSupport[si][2] = n
                            self.tidlist[si].append(n)
        for x, y in self.mapSupport.items():
            lp = abs(self.last - self.mapSupport[x][2])
            if lp <= self.maxPer:
                self.mapSupport[x][0] += 1
        self.mapSupport = {k: [v[1], v[0]] for k, v in self.mapSupport.items() if
                           v[1] >= self.minSup and v[0] / (self.minSup + 1) >= self.minPR}
        plist = [key for key, value in sorted(self.mapSupport.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        return plist

    def save(self, prefix, suffix, tidsetx):
        """
        sava prefix patterns with support and periodic ratio
        :param prefix: prefix patterns
        :type prefix: list
        :param suffix: it represents suffix itemsets
        :type suffix: list
        :param tidsetx: it represents prefix tids
        :type tidsetx: list
        """
        tidsetx = list(set(tidsetx))
        if (prefix == None):
            prefix = suffix
        else:
            prefix = prefix + suffix
        val = self.getPerSup(tidsetx)
        val1 = self.getPer_Sup(tidsetx)
        # print(prefix,tidsetx,val,val1)
        if len(tidsetx) >= self.minSup and val / (len(tidsetx) + 1) >= self.minPR:
            """self.itemsetCount+=1
            s1=str(prefix)+":"+str(len(tidsetx))+":"+str(val1)
            self.writer.write('%s \n'%s1)"""
            self.finalPatterns[tuple(prefix)] = [len(tidsetx), val1]

    def Generation(self, prefix, itemsets, tidsets):
        """
        here equibalence class is followed amd checks fro the patterns generated for periodic frequent patterns.
        :param prefix: main equivalence prefix
        :type prefix: periodic-frequent item or pattern
        :param itemsets: patterns which are items combined with prefix and satisfythe periodicity
                        and frequent wthi their time stamps
        :type itemsets: list
        :param tidsets: time stamps of the items in the argument itemSets
        :type tidsets: list
        """
        if (len(itemsets) == 1):
            i = itemsets[0]
            tidi = tidsets[0]
            self.save(prefix, [i], tidi)
            return
        for i in range(len(itemsets)):
            itemx = itemsets[i]
            if (itemx == None):
                continue
            tidsetx = tidsets[i]
            classItemsets = []
            classtidsets = []
            itemsetx = [itemx]
            for j in range(i + 1, len(itemsets)):
                itemj = itemsets[j]
                tidsetj = tidsets[j]
                y = list(set(tidsetx) & set(tidsetj))
                val = self.getPerSup(y)
                # if(len(y)>=minsup and val/(len(y)+1)>=minpr):
                if len(y) >= self.minSup and val / (self.minSup + 1) >= self.minPR:
                    classItemsets.append(itemj)
                    classtidsets.append(y)
            newprefix = list(set(itemsetx)) + prefix
            self.Generation(newprefix, classItemsets, classtidsets)
            self.save(prefix, list(set(itemsetx)), tidsetx)

    def startMine(self):
        """
        Main program start with extracting the periodic frequent items from the database and
        performs prefix equivalence to form the combinations and generates closed periodic frequent patterns.
        """
        self.path = self.iFile
        starttime = time.time()
        plist = self.scanDatabase(self.path)
        print(len(plist))
        for i in range(len(plist)):
            itemx = plist[i]
            tidsetx = self.tidlist[itemx]
            itemsetx = [itemx]
            itemsets = []
            tidsets = []
            for j in range(i + 1, len(plist)):
                itemj = plist[j]
                tidsetj = self.tidlist[itemj]
                y1 = list(set(tidsetx) & set(tidsetj))
                val = self.getPerSup(y1)
                # if(len(y1)>=minsup and val/(len(y1)+1)>=minpr):
                if len(y1) >= self.minSup and val / (self.minSup + 1) >= self.minPR:
                    itemsets.append(itemj)
                    tidsets.append(y1)
            self.Generation(itemsetx, itemsets, tidsets)
            self.save(None, itemsetx, tidsetx)
        # print("eclat Total Itemsets:",self.itemsetCount)
        endtime = time.time()
        self.runTime = (endtime - starttime)
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        # print("eclat Time taken:",temp)
        # print("eclat Memory Space:",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

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

        return self.runTime

    def getPatternsInDataFrame(self):
        """Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataframe = pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataframe

    def storePatternsInFile(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = str(x) + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPartialPeriodicPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == '__main__':
    if len(sys.argv) == 6:
        ap = PPF_DFS(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        ap.startMine()
        frequentPatterns = ap.getPartialPeriodicPatterns()
        print(f"Total number of Frequent Patterns: {len(frequentPatterns)}")
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print(f'Total Memory in USS: {memUSS}')
        memRSS = ap.getMemoryRSS()
        print(f'Total Memory in RSS: {memRSS}')
        run = ap.getRuntime()
        print(f'Total ExecutionTime in seconds: {run}')
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")


