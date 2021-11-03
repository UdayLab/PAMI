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

from abstract import *
import sys
import validators
from urllib.request import urlopen


class TopkPFPGrowth(periodicFrequentPatterns):
    """
        Top - K is and algorithm to discover top periodic frequent patterns in a temporal database.

        Reference:
        ----------
            Komate Amphawan, Philippe Lenca, Athasit Surarerks: "Mining Top-K Periodic-Frequent Pattern from Transactional Databases without Support Threshold"
            International Conference on Advances in Information Technology: https://link.springer.com/chapter/10.1007/978-3-642-10392-6_3

        Attributes:
        ----------
            iFile : str
                Input file name or path of the input file
            k: int
                User specified counte of top frequent patterns
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
            creatingItemSets()
                Scans the dataset or dataframes and stores in list format
            frequentOneItem()
                Generates one frequent patterns
            eclatGeneration(candidateList)
                It will generate the combinations of frequent items
            generateFrequentPatterns(tidList)
                It will generate the combinations of frequent items from a list of items

        Executing the code on terminal:
        -------------------------------

            Format:
            ------
                python3 FAE.py <inputFile> <outputFile> <k> <maxPer>

            Examples:
            ---------
                python3 FAE.py sampleDB.txt patterns.txt 10 3


        Sample run of the importing code:
        ---------------------------------

            import PAMI.periodicFrequentPattern.topk.TopkPFPGrowth as alg

            obj = alg.TopkPFPGrowth(iFile, k, maxPer)

            obj.startMine()

            periodicFrequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(periodicFrequentPatterns))

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

    startTime = float()
    endTime = float()
    k = int()
    maxPer = " "
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    tidList = {}
    lno = int()
    minimum = int()
    mapSupport = {}

    def creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable

        """
        self.Database = []
        if isinstance(self.iFile, pd.DataFrame):
            data, ts = [], []
            if self.iFile.empty:
                print("its empty..")
            i = self.iFile.columns.values.tolist()
            if 'TS' in i:
                ts = self.iFile['TS'].tolist()
            if 'Transactions' in i:
                data = self.iFile['Transactions'].tolist()
            if 'Patterns' in i:
                data = self.iFile['Patterns'].tolist()
            for i in range(len(data)):
                tr = [ts[i][0]]
                tr = tr + data[i]
                self.Database.append(tr)
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

    def convert(self, value):
        """
        To convert the given user specified value
        :param value: user specified value
        :return: converted value
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

    def frequentOneItem(self):
        """
        Generating one frequent patterns
        """

        self.mapSupport = {}
        self.tidList = {}
        for line in self.Database:
            self.lno += 1
            s = line
            for i in range(1, len(s)):
                si = s[i]
                if self.mapSupport.get(si) is None:
                    self.mapSupport[si] = [1, abs(0 - self.lno), self.lno]
                    self.tidList[si] = [self.lno]
                else:
                    self.mapSupport[si][0] += 1
                    self.mapSupport[si][1] = max(self.mapSupport[si][1], abs(self.lno - self.mapSupport[si][2]))
                    self.mapSupport[si][2] = self.lno
                    self.tidList[si].append(self.lno)
        for x, y in self.mapSupport.items():
            self.mapSupport[x][1] = max(self.mapSupport[x][1], abs(self.lno - self.mapSupport[x][2]))
        self.maxPer = self.convert(self.maxPer)
        self.mapSupport = {k: [v[0], v[1]] for k, v in self.mapSupport.items() if v[1] <= self.maxPer}
        plist = [key for key, value in sorted(self.mapSupport.items(), key=lambda x: (x[1][0], x[0]), reverse=True)]
        self.finalPatterns = {}
        print(len(plist))
        for i in plist:
            if len(self.finalPatterns) >= self.k:
                break
            else:
                self.finalPatterns[i] = [self.mapSupport[i][0], self.mapSupport[i][1]]
        self.minimum = min([self.finalPatterns[i][0] for i in self.finalPatterns.keys()])
        plist = list(self.finalPatterns.keys())
        return plist

    def getSupportAndPeriod(self, timeStamps):
        """To calculate the periodicity and support
        :param timeStamps: Timestamps of an item set
        :return: support, periodicity
        """

        global lno
        timeStamps.sort()
        cur = 0
        per = list()
        sup = 0
        for j in range(len(timeStamps)):
            per.append(timeStamps[j] - cur)
            cur = timeStamps[j]
            sup += 1
        per.append(self.lno - cur)
        if len(per) == 0:
            return [0, 0]
        return [sup, max(per)]

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
        sample = str()
        for i in prefix:
            sample = sample + i + " "
        if len(self.finalPatterns) < self.k:
            if val[0] >= self.minimum:
                self.finalPatterns[sample] = val
                self.finalPatterns = {k: v for k, v in
                                  sorted(self.finalPatterns.items(), key=lambda item: item[1], reverse=True)}
                self.minimum = min([self.finalPatterns[i][0] for i in self.finalPatterns.keys()])
        else:
            for x, y in sorted(self.finalPatterns.items(), key=lambda x: x[1][0]):
                if val[0] > y[0]:
                    del self.finalPatterns[x]
                    self.finalPatterns[x] = y
                    self.finalPatterns = {k: v for k, v in
                                          sorted(self.finalPatterns.items(), key=lambda item: item[1], reverse=True)}
                    self.minimum = min([self.finalPatterns[i][0] for i in self.finalPatterns.keys()])
                    return

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
            for j in range(i + 1, len(itemSets)):
                itemJ = itemSets[j]
                tidSetJ = tidSets[j]
                y = list(set(tidSetI).intersection(tidSetJ))
                val = self.getSupportAndPeriod(y)
                if val[0] >= self.minimum and val[1] <= self.maxPer:
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newPrefix = list(set(itemSetX)) + prefix
            self.Generation(newPrefix, classItemSets, classTidSets)
            self.save(prefix, list(set(itemSetX)), tidSetI)

    def startMine(self):
        """
            Main function of the program

        """
        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.k is None:
            raise Exception("Please enter the Minimum Support")
        self.creatingItemSets()
        plist = self.frequentOneItem()
        for i in range(len(plist)):
            itemI = plist[i]
            tidSetI = self.tidList[itemI]
            itemSetX = [itemI]
            itemSets = []
            tidSets = []
            for j in range(i + 1, len(plist)):
                itemJ = plist[j]
                tidSetJ = self.tidList[itemJ]
                y1 = list(set(tidSetI).intersection(tidSetJ))
                val = self.getSupportAndPeriod(y1)
                if val[0] >= self.minimum and val[1] <= self.maxPer:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self.Generation(itemSetX, itemSets, tidSets)
        print("TopK Periodic Frequent patterns were generated successfully")
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

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def savePatterns(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            patternsAndSupport = x + ":" + str(y)
            writer.write("%s \n" % patternsAndSupport)

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
            ap = TopkPFPGrowth(sys.argv[1], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 5:
            ap = TopkPFPGrowth(sys.argv[1], sys.argv[3], sys.argv[4])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Frequent Patterns:", len(Patterns))
        ap.savePatterns(sys.argv[2])
        print(ap.getPatternsAsDataFrame())
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
