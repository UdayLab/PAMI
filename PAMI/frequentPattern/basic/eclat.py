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
# from abstract import *

import sys
from abstract import *


class Eclat(frequentPatterns):
    """ Eclat is one of the fundamental algorithm to discover frequent patterns in a transactional database.
        This program employs downward closure property to  reduce the search space effectively.
        This algorithm employs depth-first search technique to find the complete set of frequent patterns in a
        transactional database.

        Reference:
        ----------
            Mohammed Javeed Zaki: Scalable Algorithms for Association Mining. IEEE Trans. Knowl. Data Eng. 12(3):
            372-390 (2000), https://ieeexplore.ieee.org/document/846291

        Attributes
        ----------
            minSup: float
                UserSpecified minimum support value. It has to be given in terms of count of total number of
                transactions in the input database/file
            startTime:float
                To record the start time of the mining process
            endTime:float
                To record the completion time of the mining process
            finalPatterns: dict
                Storing the complete set of patterns in a dictionary variable
            oFile : str
                Name of the output file to store complete set of frequent patterns
            iFile : str
                Input file name or path of the input file
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
            findDelimiter(line)
                Identifying the delimiter of the input file
            creatingItemSets(iFileName)
                Storing the complete transactions of the database/input file in a database variable
            frequentOneItem()
                Generating one frequent patterns
            dictKeysToInt(iList)
                Converting dictionary keys to integer elements
            eclatGeneration(cList)
                It will generate the combinations of frequent items
            generateFrequentPatterns(tidList)
                It will generate the combinations of frequent items from a list of items

        Executing the code on terminal:
        -------------------------------

        Format:
        ------
        python3 eclat.py <inputFile> <outputFile> <minSup>

        Examples:
        ---------
        python3 eclat.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

        python3 eclat.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)


        Sample run of the importing code:
        ---------------------------------
        import PAMI.frequentPattern.basic.eclat as alg

        obj = alg.Eclat(iFile, minSup)

        obj.startMine()

        frequentPatterns = obj.getFrequentPatterns()

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

    minSup = float()
    startTime = float()
    endTime = float()
    finalPatterns = {}
    iFile = " "
    oFile = " "
    sep = " "
    memoryUSS = float()
    memoryRSS = float()
    Database = []

    def creatingItemSets(self):
        """Storing the complete transactions of the database/input file in a database variable

            """
        try:
            with open(self.iFile, 'r', encoding='utf-8') as f:
                for line in f:
                    line.strip()
                    splitter = [i.rstrip() for i in line.split(self.sep)]
                    self.Database.append(splitter)
        except IOError:
            print("File Not Found")
            quit()

    # function to get frequent one pattern
    def frequentOneItem(self):
        """Generating one frequent patterns"""

        candidate = {}
        for i in range(len(self.Database)):
            for j in range(len(self.Database[i])):
                if self.Database[i][j] not in candidate:
                    candidate[self.Database[i][j]] = [i]
                else:
                    candidate[self.Database[i][j]] += [i]
        self.finalPatterns = {keys: value for keys, value in candidate.items() if len(value) >= self.minSup}

    @staticmethod
    def dictKeysToInt(iList):
        """Converting dictionary keys to integer elements

        :param iList: Dictionary with patterns as keys and their support count as a value
        :type iList: dict
        :returns: list of integer patterns to represent dictionary keys
        :rtype: list
        """

        temp = []
        for ite in iList.keys():
            ite = [int(i) for i in ite.strip('[]').split()]
            temp.append(ite)
            # print(sorted(temp))
        return sorted(temp)

    def eclatGeneration(self, cList):
        """It will generate the combinations of frequent items

        :param cList :it represents the items with their respective transaction identifiers
        :type cList: dictionary
        :return: returning transaction dictionary
        :rtype: dict
        """
        # to generate all
        tidList = {}
        key = list(cList.keys())
        for i in range(0, len(key)):
            for j in range(i + 1, len(key)):
                intersectionList = list(set(cList[key[i]]).intersection(set(cList[key[j]])))
                itemList = []
                itemList += key[i]
                itemList += key[j]
                if len(intersectionList) >= self.minSup:
                    itemList.sort()
                    if tuple(itemList) not in tidList:
                        tidList[tuple(set(itemList))] = intersectionList
        return tidList

    def generateFrequentPatterns(self, tidList):
        """It will generate the combinations of frequent items from a list of items

        :param tidList :it represents the items with their respective transaction identifiers
        :type tidList: dictionary
        :return: returning transaction dictionary
        :rtype: dict
        """
        tidList1 = {}
        if len(tidList) == 0:
            print("There are no more candidate sets")
        else:
            key = list(tidList.keys())
            for i in range(0, len(key)):
                for j in range(i + 1, len(key)):
                    intersectionList = list(set(tidList[key[i]]).intersection(set(tidList[key[j]])))
                    itemList = []
                    if len(intersectionList) >= self.minSup:
                        itemList += key[i], key[j]
                        itemList.sort()
                        tidList1[tuple(itemList)] = intersectionList
        return tidList1

    def convert(self, value):
        """
        To convert the user specified minSup value
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
        """Frequent pattern mining process will start from here"""

        self.startTime = time.time()
        if self.iFile is None:
            raise Exception("Please enter the file path or file name:")
        if self.minSup is None:
            raise Exception("Please enter the Minimum Support")
        self.creatingItemSets()
        self.minSup = self.convert(self.minSup)
        self.frequentOneItem()
        frequentSet = self.generateFrequentPatterns(self.finalPatterns)
        for x, y in frequentSet.items():
            if x not in self.finalPatterns:
                sample = str()
                for k in x:
                    sample = sample + k + " "
                self.finalPatterns[sample] = y
        while 1:
            frequentSet = self.eclatGeneration(frequentSet)
            for x, y in frequentSet.items():
                if x not in self.finalPatterns:
                    sample = str()
                    for k in x:
                        sample = sample + k + " "
                    self.finalPatterns[sample] = y
            if len(frequentSet) == 0:
                break
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Eclat algorithm")

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
            data.append([a, len(b)])
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
            patternsAndSupport = x + ":" + str(len(y))
            writer.write("%s \n" % patternsAndSupport)

    def getFrequentPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self.finalPatterns


if __name__ == "__main__":
    if len(sys.argv) == 4:
        ap = Eclat(sys.argv[1], sys.argv[3])
        ap.startMine()
        frequentPatterns = ap.getFrequentPatterns()
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
        
