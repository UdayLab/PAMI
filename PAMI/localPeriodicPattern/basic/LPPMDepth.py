
from PAMI.localPeriodicPattern.basic.abstract import *


class LPPMDepth(localPeriodicPatterns):

    """
    Attributes:
    -----------
        iFile : str
            Input file name or path of the input file
        oFile : str
            Output file name or path of the output file
        maxPer : float
            User defined maxPer value.
        maxSoPer : float
            User defined maxSoPer value.
        minDur : float
            User defined minDur value.
        tsmin : int / date
            First time stamp of input data.
        tsmax : int / date
            Last time stamp of input data.
        startTime : float
            Time when start of execution the algorithm.
        endTime : float
            Time when end of execution the algorithm.
        finalPatterns : dict
            To store local periodic patterns and its PTL.
        tsList : dict
            To store items and its time stamp as bit vector.
        sep : str
            separator used to distinguish items from each other. The default separator is tab space.

    Methods:
    -------
        createTSlist()
            Create the TSlist as bit vector from input data.
        generateLPP()
            Generate 1 length local periodic pattens by TSlist and execute depth first search.
        calculatePTL(tsList)
            Calculate PTL from input tsList as bit vector
        LPPMDepthSearch(extensionOfP)
            Mining local periodic patterns using depth first search.
        startMine()
            Mining process will start from here.
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function.
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function.
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function.
        getLocalPeriodicPatterns()
            return local periodic patterns and its PTL
        savePatterns(oFile)
            Complete set of local periodic patterns will be loaded in to a output file.
        getPatternsAsDataFrame()
            Complete set of local periodic patterns will be loaded in to a dataframe.

    Executing the code on terminal:
    ------------------------------
        Format:
            python3 LPPMDepth.py <inputFile> <outputFile> <maxPer> <minSoPer> <minDur> <sep>
        Examples:
            python3 LPPMDepth.py sampleDB.txt patterns.txt 0.3 0.4 0.5

            python3 LPPMDepth.py sampleDB.txt patterns.txt 3 4 5

    Sample run of importing the code:
    --------------------------------
        from PAMI.localPeriodicPattern.basic import LPPMDepth as alg

        obj = alg.LPPMDepth(iFile, maxPer, maxSoPer, minDur)

        obj.startMine()

        localPeriodicPatterns = obj.getPatterns()

        print(f'Total number of local periodic patterns: {len(localPeriodicPatterns)}')

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getMemoryUSS()

        print(f'Total memory in USS: {memUSS}')

        memRSS = obj.getMemoryRSS()

        print(f'Total memory in RSS: {memRSS}')

        runtime = obj.getRuntime()

        print(f'Total execution time in seconds: {runtime})

    Credits:
    -------
        The complete program was written by So Nakamura under the supervision of Professor Rage Uday Kiran.
    """
    
    iFile = ''
    oFile = ''
    maxPer = str()
    maxSoPer = str()
    minDur = str()
    tsmin = 0
    tsmax = 0
    startTime = float()
    endTime = float()
    memoryUSS = float()
    memoryRSS = float()
    finalPatterns = {}
    tsList = {}
    sep = ' '
    Database = []

    def creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        self.Database = []
        if isinstance(self.iFile, pd.DataFrame):
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'Transactions' in i:
                self.Database = self.iFile['Transactions'].tolist()
            if 'Patterns' in i:
                self.Database = self.iFile['Patterns'].tolist()

        if isinstance(self.iFile, str):
            if validators.url(self.iFile):
                data = urlopen(self.iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    self.Database.append(temp)
            else:
                try:
                    with open(self.iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            self.Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()

    def createTSlist(self):
        """
        Create tsList as bit vector from temporal data.
        """
        for line in self.Database:
            count = 1
            bitVector = 0b1 << count
            bitVector = bitVector | 0b1
            self.tsmin = int(line.pop(0))
            self.tsList = {item: bitVector for item in line}
            count += 1
            ts = ' '
            for line in self.Database:
                bitVector = 0b1 << count
                bitVector = bitVector | 0b1
                ts = line.pop(0)
                for item in line:
                    if self.tsList.get(item):
                        different = abs(bitVector.bit_length() - self.tsList[item].bit_length())
                        self.tsList[item] = self.tsList[item] << different
                        self.tsList[item] = self.tsList[item] | 0b1
                    else:
                        self.tsList[item] = bitVector
                count += 1
            self.tsmax = int(ts)
            for item in self.tsList:
                different = abs(bitVector.bit_length() - self.tsList[item].bit_length())
                self.tsList[item] = self.tsList[item] << different
        self.maxPer = (count-1) * self.maxPer
        self.maxSoPer = (count-1) * self.maxSoPer
        self.minDur = (count-1) * self.minDur

    def generateLPP(self):
        """
        Generate local periodic items from bit vector tsList.
        When finish generating local periodic items, execute mining depth first search.
        """
        I = set()
        PTL = {}
        for item in self.tsList:
            PTL[item] = set()
            ts = list(bin(self.tsList[item]))
            ts = ts[2:]
            start = -1
            currentTs = 1
            tsPre = ' '
            soPer = ' '
            for t in ts[currentTs:]:
                if t == '0':
                    currentTs += 1
                    continue
                else:
                    tsPre = currentTs
                    currentTs += 1
                    break
            for t in ts[currentTs:]:
                if t == '0':
                    currentTs += 1
                    continue
                else:
                    per = currentTs - tsPre
                    if per <= self.maxPer and start == -1:
                        start = tsPre
                        soPer = self.maxSoPer
                    if start != -1:
                        soPer = max(0, soPer + per - self.maxPer)
                        if soPer > self.maxSoPer:
                            if tsPre - start >= self.minDur:
                                PTL[item].add((start, tsPre))
                            """else:
                                bitVector = 0b1 << currentTs
                                different = abs(self.tsList[item].bit_length() - bitVector.bit_length())
                                bitVector = bitVector | 0b1
                                bitVector = bitVector << different
                                self.tsList[item] = self.tsList[item] | bitVector"""
                            start = -1
                    tsPre = currentTs
                    currentTs += 1
            if start != -1:
                soPer = max(0, soPer + self.tsmax - tsPre - self.maxPer)
                if soPer > self.maxSoPer and tsPre - start >= self.minDur:
                    PTL[item].add((start, tsPre))
                """else:
                    bitVector = 0b1 << currentTs+1
                    different = abs(self.tsList[item].bit_length() - bitVector.bit_length())
                    bitVector = bitVector | 0b1
                    bitVector = bitVector << different
                    self.tsList[item] = self.tsList[item] | bitVector"""
                if soPer <= self.maxSoPer and self.tsmax - start >= self.minDur:
                    PTL[item].add((start, self.tsmax))
                """else:
                    bitVector = 0b1 << currentTs+1
                    different = abs(self.tsList[item].bit_length() - bitVector.bit_length())
                    bitVector = bitVector | 0b1
                    bitVector = bitVector << different
                    self.tsList[item] = self.tsList[item] | bitVector"""
            if len(PTL[item]) > 0:
                I |= {item}
                self.finalPatterns[item] = PTL[item]
        I = sorted(list(I))
        # I = set(I)
        self.LPPMDepthSearch(I)

    def calculatePTL(self, tsList):
        """
        calculate PTL from tslist as bit vector.
        :param tsList: it is one item's tslist which is used bit vector.
        :type tsList: int
        :return: it is PTL of input item.
        """
        tsList = list(bin(tsList))
        tsList = tsList[2:]
        start = -1
        currentTs = 1
        PTL = set()
        tsPre = ' '
        soPer = ' '
        for ts in tsList[currentTs:]:
            if ts == '0':
                currentTs += 1
                continue
            else:
                tsPre = currentTs
                currentTs += 1
                break
        for ts in tsList[currentTs:]:
            if ts == '0':
                currentTs += 1
                continue
            else:
                per = currentTs - tsPre
                if per <= self.maxPer and start == -1:
                    start = tsPre
                    soPer = self.maxSoPer
                if start != -1:
                    soPer = max(0, soPer + per - self.maxPer)
                    if soPer > self.maxSoPer:
                        if tsPre - start >= self.minDur:
                            PTL.add((start, tsPre))
                        start = -1
                tsPre = currentTs
                currentTs += 1
        if start != -1:
            soPer = max(0, soPer + self.tsmax - tsPre - self.maxPer)
            if soPer > self.maxSoPer and tsPre - start >= self.minDur:
                PTL.add((start, tsPre))
            if soPer <= self.maxSoPer and self.tsmax - start >= self.minDur:
                PTL.add((start, tsPre))
        return PTL

    def LPPMDepthSearch(self, extensionsOfP):
        """
        Mining n-length local periodic pattens from n-1-length patterns by depth first search.
        :param extensionsOfP: it is n-1 length patterns list
        :type extensionsOfP: list
        """
        for x in range(len(extensionsOfP)-1):
            extensionsOfPx = set()
            for y in range(x+1,len(extensionsOfP)):
                tspxy = self.tsList[extensionsOfP[x]] & self.tsList[extensionsOfP[y]]
                PTL = self.calculatePTL(tspxy)
                if len(PTL) > 0:
                    if type(extensionsOfP[x]) == str:
                        pattern = (extensionsOfP[x], extensionsOfP[y])
                        self.finalPatterns[pattern] = PTL
                        self.tsList[pattern] = tspxy
                        extensionsOfPx.add(pattern)
                    else:
                        px = [item for item in extensionsOfP[x]]
                        py = [item for item in extensionsOfP[y]]
                        pattern = set(px + py)
                        self.finalPatterns[tuple(pattern)] = PTL
                        self.tsList[tuple(pattern)] = tspxy
                        extensionsOfPx.add(tuple(pattern))
            if extensionsOfPx:
                self.LPPMDepthSearch(list(extensionsOfPx))

    def convert(self, value):
        """
        to convert the type of user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (len(self.Database) * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (len(self.Database) * value)
            else:
                value = int(value)
        return value


    def startMine(self):
        """
        Mining process start from here. This function calls createTSlist and generateLPP.
        """
        self.startTime = time.time()
        self.finalPatterns = {}
        self.creatingItemSets()
        self.maxPer = self.convert(self.maxPer)
        self.maxSoPer = self.convert(self.maxSoPer)
        self.minDur = self.convert(self.minDur)
        self.createTSlist()
        self.generateLPP()
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryRSS = float()
        self.memoryUSS = float()
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

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
        """Storing final local periodic patterns in a dataframe

        :return: returning local periodic patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'PTL'])
        return dataFrame

    def savePatterns(self, outFile):
        """Complete set of local periodic patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            writer.write(f'{x} : {y}\n')
            # patternsAndPTL = x + ":" + y
            # writer.write("%s \n" % patternsAndPTL)

    def getPatterns(self):
        """ Function to send the set of local periodic patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == '__main__':
    ap = str()
    if len(sys.argv) == 6 or len(sys.argv) == 7:
        if len(sys.argv) == 7:
            ap = LPPMDepth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        if len(sys.argv) == 6:
            ap = LPPMDepth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Frequent Patterns:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        l = [0.004, 0.005, 0.006, 0.007, 0.008]
        for i in l:
            ap = LPPMDepth('https://www.u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv'
                             , i, 0.01, 0.01)
            ap.startMine()
            Patterns = ap.getPatterns()
            print("Total number of Frequent Patterns:", len(Patterns))
            ap.savePatterns('/Users/Likhitha/Downloads/output')
            memUSS = ap.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = ap.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = ap.getRuntime()
            print("Total ExecutionTime in ms:", run)
        print("Error! The number of input parameters do not match the total number of parameters provided")

