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

import sys
from PAMI.frequentPattern.basic.abstract import *

class eclatbitset(frequentPatterns):
    """
    EclatBitset is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs downward closure property to  reduce the search space effectively. This algorithm employs depth-first search technique to find the complete set of frequent patterns in a transactional database.

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
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
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
        python3 eclatbitset.py <inputFile> <outputFile> <minSup>

        Examples:
        ---------
        python3 eclatbitset.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

        python3 eclatbitset.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)

    Sample run of the importing code:
    ---------------------------------

        import PAMI.frequentPattern.basic.eclatbitset as alg

        obj = alg.eclatbitset(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getPatterns()

        print("Total number of Frequent Patterns:", len(frequentPatterns))

        obj.storePatternsInFile(oFile)

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
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    minSup = str()
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    mapSupport = {}
    lno = 0

    def convert(self, value):
        """
        To convert the user specified minSup value

        :param value: user specified minSup value

        :return: converted type
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
        items = []
        p = {}
        with open(self.iFile, 'r') as f:
            for line in f:
                self.lno += 1
                splitter = [i.rstrip() for i in line.split(self.sep)]
                for i in splitter:
                    if i not in items:
                        items.append(i)
        self.minSup = self.convert(self.minSup)
        with open(self.iFile, 'r') as f:
            for line in f:
                li = [i.rstrip() for i in line.split(self.sep)]
                for j in items:
                    count = 0
                    if j in li:
                        count = 1
                    if j not in p:
                        p[j] = [count]
                    else:
                        p[j].append(count)
        for x, y in p.items():
            if self.countSupport(y) >= self.minSup:
                self.mapSupport[x] = y
        pList = [key for key, value in sorted(self.mapSupport.items(), key=lambda x: (len(x[1])), reverse=True)]
        return pList

    @staticmethod
    def countSupport(tids):
        """To count support of 1's in tids

        :param tids: bitset representation of itemSets

        :return:  count
        """
        count = 0
        for i in tids:
            if i == 1:
                count += 1
        return count

    def save(self, prefix, suffix, tidSetX):
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
        count = self.countSupport(tidSetX)
        sample = str()
        for i in prefix:
            sample = sample + i + " "
        self.finalPatterns[sample] = count

    def generationOfAll(self, prefix, itemSets, tidSets):
        """It will generate the combinations of frequent items with prefix and  list of items

            :param prefix: it represents the prefix item to form the combinations

            :type prefix: list

            :param itemSets: it represents the suffix items of prefix

            :type itemSets: list

            :param tidSets: represents the tidlists of itemSets

            :type tidSets: 2d list
        """
        if len(itemSets) == 1:
            i = itemSets[0]
            tidi = tidSets[0]
            self.save(prefix, [i], tidi)
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
                support = self.countSupport(y)
                if support >= self.minSup:
                    classItemSets.append(itemJ)
                    classTidSets.append(y)
            newprefix = list(set(itemSetx)) + prefix
            self.generationOfAll(newprefix, classItemSets, classTidSets)
            del classItemSets, classTidSets
            self.save(prefix, list(set(itemSetx)), tidSetX)

    def startMine(self):
        """Frequent pattern mining process will start from here
        We start with the scanning the itemSets and store the bitsets respectively.
        We form the combinations of single items and  check with minSup condition to check the frequency of patterns
        """

        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.minSup is None:
            raise Exception("Please enter the Minimum Support")
        plist = self.creatingItemSets()
        for i in range(len(plist)):
            itemI = plist[i]
            tidSetX = self.mapSupport[itemI]
            itemSetx = [itemI]
            itemSets = []
            tidSets = []
            for j in range(i + 1, len(plist)):
                itemJ = plist[j]
                tidSetJ = self.mapSupport[itemJ]
                y1 = [k & l for k, l in zip(tidSetX, tidSetJ)]
                support = self.countSupport(y1)
                if support >= self.minSup:
                    itemSets.append(itemJ)
                    tidSets.append(y1)
            self.generationOfAll(itemSetx, itemSets, tidSets)
            del itemSets, tidSets
            self.save(None, itemSetx, tidSetX)
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Eclat_bitset algorithm")

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
            data.append([a, b])
            dataFrame = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def storePatternsInFile(self, outFile):
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
    if len(sys.argv) == 4 or len(sys.argv) == 5:
        if len(sys.argv) == 5:
            ap = eclatbitset(sys.argv[1], sys.argv[3], sys.argv[4])
        if len(sys.argv) == 4:
            ap = eclatbitset(sys.argv[1], sys.argv[3])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Frequent Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
