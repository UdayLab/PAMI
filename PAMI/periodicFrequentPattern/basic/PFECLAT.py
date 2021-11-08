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

from PAMI.periodicFrequentPattern.basic.abstract import *


class PFECLAT(periodicFrequentPatterns):
    """ EclatPFP is the fundamental approach to mine the periodic-frequent patterns.

        Reference:
        --------
            P. Ravikumar, P.Likhitha, R. Uday kiran, Y. Watanobe, and Koji Zettsu, "Towards efficient discovery of 
            periodic-frequent patterns in columnar temporal databases", 2021 IEA/AIE.

    Attributes:
    ----------
        iFile : file
            Name of the Input file or path of the input file
        oFile : file
            Name of the output file or path of the output file
        minSup: int or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        maxPer: int or float or str
            The user can specify maxPer either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        startTime:float
            To record the start time of the mining process
        endTime:float
            To record the completion time of the mining process
        Database : list
            To store the transactions of a database in list
        mapSupport : Dictionary
            To maintain the information of item and their frequency
        lno : int
            it represents the total no of transactions
        tree : class
            it represents the Tree class
        itemSetCount : int
            it represents the total no of patterns
        finalPatterns : dict
            it represents to store the patterns
        tidList : dict
            stores the timestamps of an item
        hashing : dict
            stores the patterns with their support to check for the closed property

    Methods:
    -------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to a dataframe
        getMemoryUSS()
            Total amount of USS memory consumed by the mining process will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the mining process will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the mining process will be retrieved from this function
        creatingOneItemSets()
            Scan the database and store the items with their timestamps which are periodic frequent 
        getPeriodAndSupport()
            Calculates the support and period for a list of timestamps.
        Generation()
            Used to implement prefix class equivalence method to generate the periodic patterns recursively
            
        Executing the code on terminal:
        -------
        Format:
        ------
            python3 PFPECLAT.py <inputFile> <outputFile> <minSup>

        Examples:
        --------
            python3 PFPECLAT.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

            python3 PFPECLAT.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)
        
        Sample run of the imported code:
        --------------
        
            from PAMI.periodicFrequentPattern.basic import PFPECLAT as alg

            obj = alg.PFPECLAT("../basic/sampleTDB.txt", "2", "5")

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))

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
    
    iFile = " "
    oFile = " "
    sep = " "
    dbSize = None
    Database = None
    minSup = str()
    maxPer = str()
    tidSet = set()
    finalPatterns = {}
    startTime = None
    endTime = None
    memoryUSS = float()
    memoryRSS = float()

    def getPeriodic(self, tids: set):
        tidList = list(tids)
        tidList.sort()
        tidList.append(self.dbSize)
        cur = 0
        per = 0
        for tid in tidList:
            per = max(per, tid - cur)
            if per > self.maxPer:  # early stopping
                break
            cur = tid
        return per

    def convert(self, value):
        """
        To convert the given user specified value

        :param value: user specified value
        :return: converted value
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self.dbSize * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self.dbSize * value)
            else:
                value = int(value)
        return value

    def creatingOneItemSets(self):
        """Storing the complete transactions of the database/input file in a database variable
        """
        plist = []
        Database = []
        if isinstance(self.iFile, pd.DataFrame):
            ts, data = [], []
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self.iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self.iFile['Transactions'].tolist()
            for i in range(len(data)):
                tr = [ts[i][0]]
                tr = tr + data[i]
                Database.append(tr)
        if isinstance(self.iFile, str):
            if validators.url(self.iFile):
                data = urlopen(self.iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self.sep)]
                    temp = [x for x in temp if x]
                    Database.append(temp)
            else:
                try:
                    with open(self.iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self.sep)]
                            temp = [x for x in temp if x]
                            Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()
        tid = 0
        itemsets = {}  # {key: item, value: list of tids}
        periodicHelper = {}  # {key: item, value: [period, last_tid]}
        for line in Database:
            tid = int(line[0])
            self.tidSet.add(tid)
            for item in line[1:]:
                if item in itemsets:
                    itemsets[item].add(tid)
                    periodicHelper[item][0] = max(periodicHelper[item][0],
                                                  abs(tid - periodicHelper[item][1]))  # update current max period
                    periodicHelper[item][1] = tid  # update the last tid
                else:
                    itemsets[item] = {tid}
                    periodicHelper[item] = [abs(0 - tid), tid]  # initialize helper

        # finish all items' period
        self.dbSize = len(Database)
        self.minSup = self.convert(self.minSup)
        self.maxPer = self.convert(self.maxPer)
        del Database
        for item, _ in periodicHelper.items():
            periodicHelper[item][0] = max(periodicHelper[item][0],
                                          abs(self.dbSize - periodicHelper[item][1]))  # tid of the last transaction
        candidates = []
        for item, tids in itemsets.items():
            per = periodicHelper[item][0]
            sup = len(tids)
            if sup >= self.minSup and per <= self.maxPer:
                candidates.append(item)
                self.finalPatterns[item] = [sup, per, tids]
        return candidates
    
    def generateEclat(self, candidates):
        newCandidates = []
        for i in range(0, len(candidates)):
            prefixItem = candidates[i]
            prefixItemSet = prefixItem.split()
            for j in range(i + 1, len(candidates)):
                item = candidates[j]
                itemSet = item.split()
                if prefixItemSet[:-1] == itemSet[:-1] and prefixItemSet[-1] != itemSet[-1]:
                    _value = self.finalPatterns[item][2].intersection(self.finalPatterns[prefixItem][2])
                    sup = len(_value)
                    per = self.getPeriodic(_value)
                    if sup >= self.minSup and per <= self.maxPer:
                        newItem = prefixItem + " " + itemSet[-1]
                        self.finalPatterns[newItem] = [sup, per, _value]
                        newCandidates.append(newItem)

        if len(newCandidates) > 0:
            self.generateEclat(newCandidates)
    
    def startMine(self):
        #print(f"Optimized {type(self).__name__}")
        self.startTime = time.time()
        self.finalPatterns = {}
        frequentSets = self.creatingOneItemSets()
        self.generateEclat(frequentSets)
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
        """Storing final periodic-frequent patterns in a dataframe

        :return: returning periodic-frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b[0], b[1]])
            dataframe = pd.DataFrame(data, columns=['Patterns', 'Support', 'Periodicity'])
        return dataframe

    def savePatterns(self, outFile):
        """Complete set of periodic-frequent patterns will be loaded in to a output file

        :param outFile: name of the output file
        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = x + ":" + str(y[0]) + ":" + str(y[1])
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of periodic-frequent patterns after completion of the mining process

        :return: returning periodic-frequent patterns
        :rtype: dict
        """
        return self.finalPatterns
                    

if __name__ == "__main__":
    ap = str()
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        if len(sys.argv) == 6:
            ap = PFECLAT(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = PFECLAT(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Periodic-Frequent Patterns:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
