# Parallel Apriori is an algorithm to discover frequent patterns in a transactional database. This program employs parallel apriori property (or downward closure property) to  reduce the search space effectively.
#
#
#  **Importing this algorithm into a python program**
#  ---------------------------------------------------
#
#         import PAMI.frequentPattern.pyspark.parallelApriori as alg
#
#         obj = alg.parallelApriori(iFile, minSup, numWorkers)
#
#         obj.startMine()
#
#         frequentPatterns = obj.getPatterns()
#
#         print("Total number of Frequent Patterns:", len(frequentPatterns))
#
#         obj.save(oFile)
#
#         Df = obj.getPatternInDataFrame()
#
#         memUSS = obj.getMemoryUSS()
#
#         print("Total Memory in USS:", memUSS)
#
#         memRSS = obj.getMemoryRSS()
#
#         print("Total Memory in RSS", memRSS)
#
#         run = obj.getRuntime()
#
#         print("Total ExecutionTime in seconds:", run)
#
#
#
__copyright__ = """
 Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from PAMI.frequentPattern.pyspark import abstract as _ab


class parallelApriori(_ab._frequentPatterns):
    """

    :Description: Parallel Apriori is an algorithm to discover frequent patterns in a transactional database. This program employs parallel apriori property (or downward closure property) to  reduce the search space effectively.

    :Reference: N. Li, L. Zeng, Q. He and Z. Shi, "Parallel Implementation of Apriori Algorithm Based on MapReduce,"
                2012 13th ACIS International Conference on Software Engineering, Artificial Intelligence,
                Networking and Parallel/Distributed Computing, Kyoto, Japan, 2012, pp. 236-241, doi: 10.1109/SNPD.2012.31.

    :param  iFile: str :
                   Name of the Input file to mine complete set of frequent patterns
    :param  oFile: str :
                   Name of the output file to store complete set of frequent patterns
    :param  minSup: int :
                   The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
    :param  sep: str :
                   This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.
    :param  numPartitions: int :
                   The number of partitions. On each worker node, an executor process is started and this process performs processing.The processing unit of worker node is partition



    :Attributes:

        startTime : float
          To record the start time of the mining process

        endTime : float
          To record the completion time of the mining process

        finalPatterns : dict
          Storing the complete set of patterns in a dictionary variable

        memoryUSS : float
          To store the total amount of USS memory consumed by the program

        memoryRSS : float
          To store the total amount of RSS memory consumed by the program

        lno : int
                the number of transactions

    

    **Methods to execute code on terminal**
    -----------------------------------------
    
            Format:
                      >>>  python3 parallelApriori.py <inputFile> <outputFile> <minSup> <numWorkers>
    
            Example:
                      >>>  python3 parallelApriori.py sampleDB.txt patterns.txt 10.0 3
    
            .. note:: minSup will be considered in percentage of database transactions
    
    
    **Importing this algorithm into a python program**
    ----------------------------------------------------------------------------------
    .. code-block:: python
    
                import PAMI.frequentPattern.pyspark.parallelApriori as alg
    
                obj = alg.parallelApriori(iFile, minSup, numWorkers)
    
                obj.startMine()
    
                frequentPatterns = obj.getPatterns()
    
                print("Total number of Frequent Patterns:", len(frequentPatterns))
    
                obj.save(oFile)
    
                Df = obj.getPatternInDataFrame()
    
                memUSS = obj.getMemoryUSS()
    
                print("Total Memory in USS:", memUSS)
    
                memRSS = obj.getMemoryRSS()
    
                print("Total Memory in RSS", memRSS)
    
                run = obj.getRuntime()
    
                print("Total ExecutionTime in seconds:", run)
    
    
    **Credits:**
    -----------------------------------------
            The complete program was written by Yudai Masu  under the supervision of Professor Rage Uday Kiran.


    """

    _minSup = float()
    _startTime = float()
    _endTime = float()
    _finalPatterns = {}
    _iFile = " "
    _oFile = " "
    _sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _numPartitions = int()
    _lno = int()

    def __init__(self, iFile, minSup, numWorkers, sep='\t'):
        super().__init__(iFile, minSup, int(numWorkers), sep)

    def _creatingItemSets(self):
        """
            Storing the complete transactions of the database/input file in a database variable


        """
        self._Database = []
        if isinstance(self._iFile, _ab._pd.DataFrame):
            if self._iFile.empty:
                print("its empty..")
            i = self._iFile.columns.values.tolist()
            if 'Transactions' in i:
                self._Database = self._iFile['Transactions'].tolist()
        if isinstance(self._iFile, str):
            if _ab._validators.url(self._iFile):
                data = _ab._urlopen(self._iFile)
                for line in data:
                    line.strip()
                    line = line.decode("utf-8")
                    temp = [i.rstrip() for i in line.split(self._sep)]
                    temp = [x for x in temp if x]
                    self._Database.append(temp)
            else:
                try:
                    with open(self._iFile, 'r', encoding='utf-8') as f:
                        for line in f:
                            line.strip()
                            temp = [i.rstrip() for i in line.split(self._sep)]
                            temp = [x for x in temp if x]
                            #print(line)
                            self._Database.append(temp)
                except IOError:
                    print("File Not Found")
                    quit()
    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function
        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function
        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process
        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe
        :return: returning frequent patterns in a dataframe
        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a, b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        return dataFrame

    def save(self, outFile):
        """
        Complete set of frequent patterns will be loaded in to an output file
        :param outFile: name of the output file
        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = str(x) + " : " + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """
        Function to send the set of frequent patterns after completion of the mining process
        :return: returning frequent patterns
        :rtype: dict
        """
        return self._finalPatterns
    
    def printResults(self):
        """
        This method prints all the statistics
        """
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())

    @staticmethod
    def _Mapper(transaction, candidateItemsets):
        """
        Map each candidate itemset of candidateItemsets to (itemset,1) if a candidate itemset is in transaction

        :param transaction: a transaction of database
        :type transaction: set
        :param candidateItemsets: candidate item sets
        :type candidateItemsets: list
        :return:set
        """

        candidates = set()
        for itemset in candidateItemsets:
            if set(itemset).issubset(transaction):
                candidates.add((itemset, 1))
        return candidates

    @staticmethod
    def _genCandidateItemsets(frequentPatterns, length):
        """
        Generate candidate itemsets from frequentPatterns

        :param frequentPatterns: set of all frequent patterns to generate candidate patterns of each of size is length
        :type frequentPatterns: list
        :param length: size of each candidate patterns to be generated
        :type length: int
        :return: list of candidate patterns
        :rtype: list
        """
        candidates = list(_ab._c(frequentPatterns, 2))
        candidates = set([tuple(set(item[0]).union(set(item[1]))) for item in [x for x in candidates]])
        candidates = list({item for item in candidates if len(item) == length})
        return candidates

    def _genFrequentItems(self, database):
        """
        Get frequent items which length is 1


        :return: frequent items which length is 1
        :rtype: dict
        """
        frequentItems = dict(database.flatMap(lambda x: [(item, 1) for item in x])
                             .reduceByKey(lambda x, y: x + y)
                             .filter(lambda c: c[1] >= self._minSup)
                             .collect())
        return frequentItems

    def _getAllFrequentPatterns(self, database, frequentItems):
        """
        Get all frequent patterns and save them to self.oFile

        :param database: database
        :type : RDD
        :param frequentItems: dict
        :return:
        """

        # Get candidate patterns that length is 2
        candidates = list(_ab._c(frequentItems.keys(), 2))
        length = 3
        while len(candidates) != 0:
            # if each itemset of candidates is in each transaction, then create (itemset,1)
            mappedDatabase = database.flatMap(lambda transaction: self._Mapper(transaction, candidates))

            # aggregate the values by key by reduceByKey() method
            frequentPatterns = dict(
                mappedDatabase.reduceByKey(lambda x, y: x + y).filter(lambda c: c[1] >= self._minSup).collect())
            self._finalPatterns.update(frequentPatterns)
            candidates = self._genCandidateItemsets(list(frequentPatterns.keys()), length)
            length += 1

    def _convert(self, value):
        """
        To convert the user specified minSup value
        :param value: user specified minSup value
        :return: converted type
        """
        if type(value) is int:
            value = int(value)
        if type(value) is float:
            value = (self._lno * value)
        if type(value) is str:
            if '.' in value:
                value = float(value)
                value = (self._lno * value)
            else:
                value = int(value)
        return value

    def startMine(self):
        """
        Frequent pattern mining process will start from here

        :return:
        """
        self._startTime = _ab._time.time()

        # setting SparkConf and SparkContext to process in parallel
        conf = _ab._SparkConf().setAppName("parallelApriori").setMaster("local[*]")
        sc = _ab._SparkContext(conf=conf)
        # sc.addFile("file:///home/hadoopuser/Spark_code/abstract.py")

        # read database from iFile
        database = sc.textFile(self._iFile, self._numPartitions).map(
            lambda x: {int(y) for y in x.rstrip().split(self._sep)})
        self._lno = database.count()
        # Calculating minSup as a percentage
        self._minSup = self._convert(self._minSup)

        oneFrequentItems = self._genFrequentItems(database)
        self._finalPatterns = oneFrequentItems
        self._getAllFrequentPatterns(database, oneFrequentItems)

        self._endTime = _ab._time.time()
        process = _ab._psutil.Process(_ab._os.getpid())
        self._memoryUSS = process.memory_full_info().uss
        self._memoryRSS = process.memory_info().rss
        print("Frequent patterns were generated successfully using Parallel Apriori algorithm")
        sc.stop()


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = parallelApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4], _ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = parallelApriori(_ab._sys.argv[1], _ab._sys.argv[3], _ab._sys.argv[4])
        _ap.startMine()
        _finalPatterns = _ap.getPatterns()
        print("Total number of Frequent Patterns:", len(_finalPatterns))
        _ap.savePatterns(_ab._sys.argv[2])
        _memUSS = _ap.getMemoryUSS()
        print("Total Memory in USS:", _memUSS)
        _memRSS = _ap.getMemoryRSS()
        print("Total Memory in RSS", _memRSS)
        _run = _ap.getRuntime()
        print("Total ExecutionTime in ms:", _run)
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")

