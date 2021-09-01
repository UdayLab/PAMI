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

from PAMI.frequentPattern.basic.abstract import *
import sys


class Apriori(frequentPatterns):
    """
        Apriori is one of the fundamental algorithm to discover frequent patterns in a transactional database.
        This program employs apriori property (or downward closure property) to  reduce the search space effectively.
        This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a
        transactional database.

        Reference:
        ----------
            Agrawal, R., Imieli ́nski, T., Swami, A.: Mining association rules between sets of items in large databases.
            In: SIGMOD. pp. 207–216 (1993), https://doi.org/10.1145/170035.170072


        Attributes:
        ----------
            iFile : str
                Input file name or path of the input file
            oFile : str
                Name of the output file or the path of output file
            minSup: float or int or str
                The user can specify minSup either in count or proportion of database size.
                If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
                Otherwise, it will be treated as float.
                Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
            sep : str
                This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
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
            Database : list
                To store the transactions of a database in list


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
            candidateToFrequent(candidateList)
                Generates frequent patterns from the candidate patterns
            frequentToCandidate(frequentList, length)
                Generates candidate patterns from the frequent patterns
        
        
        Executing the code on terminal:
        -------------------------------

            Format:
            ------
                python3 apriori.py <inputFile> <outputFile> <minSup>

            Examples:
            ---------
                python3 apriori.py sampleDB.txt patterns.txt 10.0   (minSup will be considered in percentage of database transactions)

                python3 apriori.py sampleDB.txt patterns.txt 10     (minSup will be considered in support count or frequency)

        Sample run of the importing code:
        ---------------------------------


            import PAMI.frequentPattern.basic.Apriori as alg

            obj = alg.Apriori(iFile, minSup)

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

    def candidateToFrequent(self, candidateList):
        """Generates frequent patterns from the candidate patterns

        :param candidateList: Candidate patterns will be given as input

        :type candidateList: list

        :return: returning set of all frequent patterns

        :rtype: dict
        """

        candidateToFrequentList = {}
        for i in self.Database:
            dictionary = {frozenset(j): int(candidateToFrequentList.get(frozenset(j), 0)) + 1 for j in candidateList if
                          j.issubset(i)}
            candidateToFrequentList.update(dictionary)
        candidateToFrequentList = {key: value for key, value in candidateToFrequentList.items() if value >= self.minSup}

        return candidateToFrequentList

    @staticmethod
    def frequentToCandidate(frequentList, length):
        """Generates candidate patterns from the frequent patterns

        :param frequentList: set of all frequent patterns to generate candidate patterns of each of size is length

        :type frequentList: dict

        :param length: size of each candidate patterns to be generated

        :type length: int

        :return: set of candidate patterns in sorted order

        :rtype: list
        """

        frequentToCandidateList = []
        for i in frequentList:
            nextList = [i | j for j in frequentList if len(i | j) == length and (i | j) not in frequentToCandidateList]
            frequentToCandidateList.extend(nextList)
        return sorted(frequentToCandidateList)

    def startMine(self):
        """
            Frequent pattern mining process will start from here
        """

        self.startTime = time.time()
        try:
            with open(self.iFile, 'r') as f:
                self.Database = [set([i.rstrip() for i in line.split(self.sep)]) for line in f]
                f.close()
        except IOError:
            print("File Not Found")
            quit()
        itemsList = sorted(list(set.union(*self.Database)))  # because Database is list
        items = [{i} for i in itemsList]
        itemsCount = len(items)
        self.minSup = self.convert(self.minSup)
        for i in range(1, itemsCount):
            frequentSet = self.candidateToFrequent(items)
            for x,y in frequentSet.items():
                sample = str()
                for k in x:
                    sample = sample + k + " "
                self.finalPatterns[sample] = y
            items = self.frequentToCandidate(frequentSet, i + 1)
            if len(items) == 0:
                break  # finish apriori
        self.endTime = time.time()
        process = psutil.Process(os.getpid())
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Apriori algorithm ")

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
            ap = Apriori(sys.argv[1], sys.argv[3], sys.argv[4])
        if len(sys.argv) == 4:
            ap =Apriori(sys.argv[1], sys.argv[3])
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
