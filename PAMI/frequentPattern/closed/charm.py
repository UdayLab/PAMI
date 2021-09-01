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
from PAMI.frequentPattern.closed.abstract import *


class Charm(frequentPatterns):
    """ CHARM is one of the fundamental algorithm to discover closed frequent patterns in a transactional database.
        Closed frequent patterns are patterns if there exists no superset that has the same support count as this original itemset.
        This program employs downward closure property to  reduce the search space effectively.
        This algorithm employs depth-first search technique to find the complete set of closed frequent patterns in a
        transactional database.
        
        Reference:
        ----------
            Mohammed J. Zaki and Ching-Jui Hsiao, CHARM: An Efficient Algorithm for Closed Itemset Mining,
            Proceedings of the 2002 SIAM, SDM. 2002, 457-473, https://doi.org/10.1137/1.9781611972726.27

       Attributes
       ----------
            iFile : file
                Name of the Input file or path of the input file
            oFile : file
                Name of the output file or path of the output file
            minSup: float or int or str
                The user can specify minSup either in count or proportion of database size.
                If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                Otherwise, it will be treated as float.
                Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
            sep : str
                This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
                However, the users can override their default separator.
            memoryUSS : float
                To store the total amount of USS memory consumed by the program
            memoryRSS : float
                To store the total amount of RSS memory consumed by the program
            startTime: float
                To record the start time of the mining process
            endTime: float
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
                Complete set of frequent patterns will be loaded in to a output file
            getPatternsInDataFrame()
                Complete set of frequent patterns will be loaded in to a dataframe
            getMemoryUSS()
                Total amount of USS memory consumed by the mining process will be retrieved from this function
            getMemoryRSS()
                Total amount of RSS memory consumed by the mining process will be retrieved from this function
            getRuntime()
                Total amount of runtime taken by the mining process will be retrieved from this function
            creatingItemsets()
                Stores the frequent patterns with their timestamps from the dataset
        
        
        Executing the code on terminal:
        -------
            Format: python3 charm.py <inputFile> <outputFile> <minSup>

            Examples:

            python3 charm.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

            python3 charm.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)

        Sample run of the importing code:
        --------------

            from PAMI.frequentPattern.closed import closed as alg

            obj = alg.Closed(iFile, minSup)

            obj.startMine()

            frequentPatterns = obj.getPatterns()

            print("Total number of Frequent Patterns:", len(frequentPatterns))

            obj.storePatternsInFile(oFile)

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
    minSup = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []
    tidList = {}
    lno = 0
    mapSupport = {}
    hashing = {}
    itemSetCount = 0
    maxItemId = 0
    tableSize = 10000
    writer = None

    def convert(self, value):
        """
        to convert the type of user specified minSup value

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

    def creatingItemsets(self):
        """
        Storing the complete frequent patterns of the database/input file in a database variable
        """
        with open(self.iFile, 'r') as f:
            for line in f:
                i = [i.rstrip() for i in line.split(self.sep)]
                self.lno += 1
                n = self.lno
                for j in i:
                    if j not in self.mapSupport:
                        self.mapSupport[j] = 1
                        self.tidList[j] = [n]
                    else:
                        self.mapSupport[j] += 1
                        self.tidList[j].append(n)
        self.minSup = self.convert(self.minSup)
        self.mapSupport = {k: v for k, v in self.mapSupport.items() if v >= self.minSup}
        flist = {}
        self.tidList = {k: v for k, v in self.tidList.items() if k in self.mapSupport}
        for x, y in self.tidList.items():
            t1 = 0
            for i in y:
                t1 += i
            flist[x] = t1
        flist = [key for key, value in sorted(flist.items(), key=lambda x: x[1])]
        return flist

    def calculate(self, tidSet):
        """To calculate the hashcode of pattern

            :param tidSet: the timestamps of a pattern

            :type tidSet: list

            :rtype: int
        """

        hashcode = 0
        for i in tidSet:
            hashcode += i
        if hashcode < 0:
            hashcode = abs(0 - hashcode)
        return hashcode % self.tableSize

    def contains(self, itemSet, value, hashcode):
        """ Check for the closed property(patterns with same support) by checking the hashcode(sum of timestamps),
            if hashcode key in hashing dict is none then returns a false, else returns with true.

            :param itemSet: frequent pattern

            :type itemSet: list

            :param value: support of the pattern

            :type value: int

            :param hashcode: calculated from the timestamps of pattern

            :type hashcode: int
            """
        if self.hashing.get(hashcode) is None:
            return False
        for i in self.hashing[hashcode]:
            itemSetx = i
            if value == self.hashing[hashcode][itemSetx] and set(itemSetx).issuperset(itemSet):
                return True
        return False

    def save(self, prefix, suffix, tidSetx):
        """ Check for the closed property (patterns with same support), if found deletes the subsets and stores
            supersets and also saves the patterns that satisfy the closed property

            :param prefix: the prefix of a pattern

            :param suffix: the suffix of a patterns

            :type suffix: list

            :param tidSetx: the timestamp of a patterns

            :type tidSetx: list
        """
        if prefix is None:
            prefix = suffix
        else:
            prefix = prefix + suffix
        prefix = list(set(prefix))
        prefix.sort()
        val = len(tidSetx)
        if val >= self.minSup:
            hashcode = self.calculate(tidSetx)
            if self.contains(prefix, val, hashcode) is False:
                sample = str()
                for i in prefix:
                    sample = sample + i + " "
                self.itemSetCount += 1
                self.finalPatterns[sample] = val
            if hashcode not in self.hashing:
                self.hashing[hashcode] = {tuple(prefix): val}
            else:
                self.hashing[hashcode][tuple(prefix)] = val

    def processEquivalenceClass(self, prefix, itemSets, tidSets):
        """ Equivalence class is followed  and check for the patterns which satisfies frequent properties.

            :param prefix:  main equivalence prefix

            :type prefix: frequent item or pattern

            :param itemSets: patterns which are items combined with prefix and satisfying the minSup

            :type itemSets: list

            :param tidSets: timestamps of the items in the argument itemSets

            :type tidSets: list


        """
        if len(itemSets) == 1:
            i = itemSets[0]
            tidI = tidSets[0]
            self.save(prefix, [i], tidI)
            return
        if len(itemSets) == 2:
            itemX = itemSets[0]
            tidSetX = tidSets[0]
            itemY = itemSets[1]
            tidSetY = tidSets[1]
            y1 = list(set(tidSetX).intersection(tidSetY))
            if len(y1) >= self.minSup:
                suffix = []
                suffix += [itemX, itemY]
                suffix = list(set(suffix))
                self.save(prefix, suffix, y1)
            if len(y1) != len(tidSetX):
                self.save(prefix, [itemX], tidSetX)
            if len(y1) != len(tidSetY):
                self.save(prefix, [itemX], tidSetY)
            return
        for i in range(len(itemSets)):
            itemX = itemSets[i]
            if itemX is None:
                continue
            tidSetX = tidSets[i]
            classItemSets = []
            classTidSets = []
            itemSetx = [itemX]
            for j in range(i + 1, len(itemSets)):
                itemY = itemSets[j]
                if itemY is None:
                    continue
                tidSetY = tidSets[j]
                y = list(set(tidSetX).intersection(tidSetY))
                if len(y) < self.minSup:
                    continue
                if len(tidSetX) == len(tidSetY) and len(y) == len(tidSetX):
                    itemSets.insert(j, None)
                    tidSets.insert(j, None)
                    itemSetx.append(itemY)
                elif len(tidSetX) < len(tidSetY) and len(y) == len(tidSetX):
                    itemSetx.append(itemY)
                elif len(tidSetX) > len(tidSetY) and len(y) == len(tidSetY):
                    itemSets.insert(j, None)
                    tidSets.insert(j, None)
                    classItemSets.append(itemY)
                    classTidSets.append(y)
                else:
                    classItemSets.append(itemY)
                    classTidSets.append(y)
            if len(classItemSets) > 0:
                newPrefix = list(set(itemSetx)) + prefix
                self.processEquivalenceClass(newPrefix, classItemSets, classTidSets)
                self.save(prefix, list(set(itemSetx)), tidSetX)

    def startMine(self):
        """
        Mining process will start from here by extracting the frequent patterns from the database. It performs prefix
        equivalence to generate the combinations and generates closed frequent patterns.
        """
        self.startTime = time.time()
        plist = self.creatingItemsets()
        for i in range(len(plist)):
            itemX = plist[i]
            if itemX is None:
                continue
            tidSetx = self.tidList[itemX]
            itemSetx = [itemX]
            itemSets = []
            tidSets = []
            for j in range(i + 1, len(plist)):
                itemY = plist[j]
                if itemY is None:
                    continue
                tidSetY = self.tidList[itemY]
                y1 = list(set(tidSetx).intersection(tidSetY))
                if len(y1) < self.minSup:
                    continue
                if len(tidSetx) == len(tidSetY) and len(y1) == len(tidSetx):
                    plist.insert(j, None)
                    itemSetx.append(itemY)
                elif len(tidSetx) < len(tidSetY) and len(y1) == len(tidSetx):
                    itemSetx.append(itemY)
                elif len(tidSetx) > len(tidSetY) and len(y1) == len(tidSetY):
                    plist.insert(j, None)
                    itemSets.append(itemY)
                    tidSets.append(y1)
                else:
                    itemSets.append(itemY)
                    tidSets.append(y1)
            if len(itemSets) > 0:
                self.processEquivalenceClass(itemSetx, itemSets, tidSets)
            self.save(None, itemSetx, tidSetx)
        print("Closed Frequent patterns were generated successfully using charm algorithm")
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
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataframe = {}
        data = []
        for a, b in self.finalPatterns.items():
            data.append([a, b])
            dataframe = pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataframe

    def storePatternsInFile(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self.oFile = outFile
        writer = open(self.oFile, 'w+')
        for x, y in self.finalPatterns.items():
            s1 = x + ":" + str(y)
            writer.write("%s \n" % s1)

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
            ap = Charm(sys.argv[1], sys.argv[3], sys.argv[4])
        if len(sys.argv) == 4:
            ap = Charm(sys.argv[1], sys.argv[3])
        ap.startMine()
        Patterns = ap.getPatterns()
        print("Total number of Closed Frequent Patterns:", len(Patterns))
        ap.storePatternsInFile(sys.argv[2])
        memUSS = ap.getMemoryUSS()
        print("Total Memory in USS:", memUSS)
        memRSS = ap.getMemoryRSS()
        print("Total Memory in RSS", memRSS)
        run = ap.getRuntime()
        print("Total ExecutionTime in ms:", run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
