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

import sys
from PAMI.periodicFrequentPattern.basic.abstract import *


class PFEclat(periodicFrequentPatterns):
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
        storePatternsInFile(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsInDataFrame()
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
        python3 PFEclat.py <inputFile> <outputFile> <minSup>

        Examples:
        --------
        python3 PFEclat.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

        python3 PFEclat.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)
        
        Sample run of the imported code:
        --------------
        
            from PAMI.periodicFrequentPattern.basic import PFEclat as alg

            obj = alg.PFEclat("../basic/sampleTDB.txt", "2", "5")

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of Periodic Frequent Patterns:", len(periodicFrequentPatterns))

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
    
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    mapSupport = {}
    hashing = {}
    itemSetCount = 0
    writer = None
    minSup = float()
    maxPer = float()
    tidList = {}
    lno = 0

    def getSupportAndPeriod(self, tids):
        """calculates the support and periodicity with list of timestamps

            :param tids: timestamps of a pattern
            :type tids: list
        """
        tids.sort()
        cur = 0
        per = 0
        sup = 0
        for j in range(len(tids)):
            per = max(per, tids[j] - cur)
            if per > self.maxPer:
                return [0, 0]
            cur = tids[j]
            sup += 1
        per = max(per, self.lno - cur)
        return [sup, per]

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

    def creatingOneItemSets(self):
        """Storing the complete transactions of the database/input file in a database variable
        """

        with open(self.iFile, 'r') as f:
            for line in f:
                self.lno += 1
                s = [i.strip() for i in line.split(self.sep)]
                n = self.lno
                for i in range(1, len(s)):
                    si = s[i]
                    if self.mapSupport.get(si) is None:
                        self.mapSupport[si] = [1, abs(0-n), n]
                        self.tidList[si] = [n]
                    else:
                        self.mapSupport[si][0] += 1
                        self.mapSupport[si][1] = max(self.mapSupport[si][1], abs(n-self.mapSupport[si][2]))
                        self.mapSupport[si][2] = n
                        self.tidList[si].append(n)
        for x, y in self.mapSupport.items():
            self.mapSupport[x][1] = max(self.mapSupport[x][1], abs(self.lno - self.mapSupport[x][2]))
        self.minSup = self.convert(self.minSup)
        self.maxPer = self.convert(self.maxPer)
        self.mapSupport = {k: [v[0], v[1]] for k, v in self.mapSupport.items() if v[0] >= self.minSup and v[1] <=
                           self.maxPer}
        plist = [key for key, value in sorted(self.mapSupport.items(), key=lambda x:(x[1][0], x[0]), reverse=True)]
        return plist
    
    def save(self, prefix, suffix, tidSetI):
        """Saves the patterns that satisfy the periodic frequent property.

            :param prefix: the prefix of a pattern
            :type prefix: list
            :param suffix: the suffix of a patterns
            :type suffix: list
            :param tidSetI: the timestamp of a patterns
            :type tidSetI: list
        """

        if prefix is None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        val = self.getSupportAndPeriod(tidSetI)
        if val[0] >= self.minSup and val[1] <= self.maxPer:
            sample = str()
            for i in prefix:
                sample = sample + i + " "
            self.finalPatterns[sample] = val
    
    def Generation(self, prefix, itemSets, tidSets):
        """Equivalence class is followed  and checks for the patterns generated for periodic-frequent patterns.

            :param prefix:  main equivalence prefix
            :type prefix: periodic-frequent item or pattern
            :param itemSets: patterns which are items combined with prefix and satisfying the periodicity
                            and frequent with their timestamps
            :type itemSets: list
            :param tidSets: timestamps of the items in the argument itemSets
            :type tidSets: list


                    """
        if len(itemSets) == 1:
            i = itemSets[0]
            tidI = tidSets[0]
            self.save(prefix, [i], tidI)
            return
        for i in range(len(itemSets)):
            itemI = itemSets[i]
            if itemI is None:
                continue
            tidSetI = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetX = [itemI]
            for j in range(i+1, len(itemSets)):
                itemJ = itemSets[j]
                tidSetJ = tidSets[j]
                y = list(set(tidSetI).intersection(tidSetJ))
                if len(y) >= self.minSup:
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newPrefix = list(set(itemSetX)) + prefix
            self.Generation(newPrefix, classItemSets, classTidSets)
            self.save(prefix, list(set(itemSetX)), tidSetI)
        
    def startMine(self):
        """ Main program start with extracting the periodic frequent items from the database and performs prefix
        equivalence to form the combinations and generates closed periodic-frequent patterns.
        """

        self.startTime = time.time()
        plist = self.creatingOneItemSets()
        for i in range(len(plist)):
            itemI = plist[i]
            tidSetI = self.tidList[itemI]
            itemSetX = [itemI]
            itemSets = []
            tidSets = []
            for j in range(i+1, len(plist)):
                itemJ = plist[j]
                tidSetJ = self.tidList[itemJ]
                y1 = list(set(tidSetI).intersection(tidSetJ))
                if len(y1) >= self.minSup:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self.Generation(itemSetX, itemSets, tidSets)
            self.save(None, itemSetX, tidSetI)
        print("Periodic-Frequent patterns were generated successfully using eclat_pfp algorithm")
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
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

    def getPatternsInDataFrame(self):
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

    def storePatternsInFile(self, outFile):
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
            ap = PFEclat(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = PFEclat(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Periodic-Frequent Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
