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

import os
import csv
import time
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import psutil


class cudaEclatGCT:
    """
            :Description: Apriori is one of the fundamental algorithm to discover frequent patterns in a transactional database. This program employs apriori property (or downward closure property) to  reduce the search space effectively. This algorithm employs breadth-first search technique to find the complete set of frequent patterns in a transactional database.

            :Reference:  Agrawal, R., Imieli ́nski, T., Swami, A.: Mining association rules between sets of items in large databases.
                    In: SIGMOD. pp. 207–216 (1993), https://doi.org/10.1145/170035.170072

            :param  iFile: str :
                           Name of the Input file to mine complete set of frequent patterns
            :param  oFile: str :
                           Name of the output file to store complete set of frequent patterns
            :param  minSup: int :
                           The user can specify minSup either in count or proportion of database size. If the program detects the data type of minSup is integer, then it treats minSup is expressed in count. Otherwise, it will be treated as float.
            :param  sep: str :
                           This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.



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

                Database : list
                  To store the transactions of a database in list



            **Methods to execute code on terminal**
            ----------------------------------------------------

                    Format:
                              >>> python3 cudaEclatGCT.py <inputFile> <outputFile> <minSup>

                    Example:
                              >>>  python3 cudaEclatGCT.py sampleDB.txt patterns.txt 10.0

                    .. note:: minSup will be considered in percentage of database transactions


            **Importing this algorithm into a python program**
            ----------------------------------------------------

            .. code-block:: python

                     import PAMI.frequentPattern.cuda.cuAprioriBit as alg

                     obj = alg.cuAprioriBit(iFile, minSup)

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
            -------------

                     The complete program was written by Tarun Sreepada under the supervision of Professor Rage Uday Kiran.

            """
    
    __time = 0
    __memRSS = 0
    __memUSS = 0
    __GPU_MEM = 0
    filePath = ""
    sep = ""
    minSup = 0
    Patterns = {}

    def __init__(self, filePath, sep, minSup):
        self.filePath = filePath
        self.sep = sep
        self.minSup = minSup
        self.__time = 0
        self.__memRSS = 0
        self.__memUSS = 0

    def read_data(self, data_path, sep):
        """
                param data_path:
                type data_path:
                param sep:
                type sep:
                """
        data = []
        if not os.path.isfile(data_path):
            raise ValueError('Invalid data path.' + data_path)
        with open(data_path, 'r') as f:
            file = csv.reader(f, delimiter=sep, quotechar='\r')
            lineNo = 1
            for row in file:
                data.append([str(item) for item in row if item != ''])
                lineNo += 1
        return data, lineNo

    def write_result(self, result, result_path):
        """
                param result:
                type result:
                param result_path:
                type result_path:
                """
        file = open(result_path, 'w')
        for itemset, support in result.items():
            file.write(str(itemset) + ' : ' + str(support) + '\n')
        file.close()

    def compute_vertical_bitvector_data(self, data):
        """
                param data:
                type data:

                """
        #---build item to idx mapping---#
        idx = 0
        item2idx = {}
        for transaction in data:
            for item in transaction:
                if not item in item2idx:
                    item2idx[item] = idx
                    idx += 1
        idx2item = {idx: str(int(item)) for item, idx in item2idx.items()}
        #---build vertical data---#
        vb_data = np.zeros((len(item2idx), len(data)), dtype=np.uint16)
        for trans_id, transaction in enumerate(data):
            for item in transaction:
                vb_data[item2idx[item], trans_id] = 1
        vb_data = gpuarray.to_gpu(vb_data.astype(np.uint16))
        return vb_data, idx2item

    def get_time(self):
        """Calculating the total amount of time taken by the mining process
                :return: returning total amount of runtime taken by the mining process
                :rtype: float
        """
        return self.__time

    def get_memRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

                            :return: returning RSS memory consumed by the mining process

                            :rtype: float
        """
        return self.__memRSS

    def get_memUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

             :return: returning USS memory consumed by the mining process

             :rtype: float
        """
        return self.__memUSS

    def get_GPU_MEM(self):

        return self.__GPU_MEM

    def get_Patterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

                :return: returning frequent patterns

                :rtype: dict
        """
        return self.Patterns

    def get_numberOfPatterns(self):
        return len(self.Patterns)

    def eclat(self, basePattern, final, vb_data, idx2item, item2idx):
        """ param basePattern:
            type basePattern:
            param final:
            type final:
            param vb_data:
            type vb_data:

        """
        newBasePattern = []
        for i in range(0, len(basePattern)):
            item1 = basePattern[i]
            i1_list = item1.split()
            for j in range(i + 1, len(basePattern)):
                item2 = basePattern[j]
                i2_list = item2.split()
                if i1_list[:-1] == i2_list[:-1]:
                    unionOfKey = list(set(i1_list) | set(i2_list))
                    unionOfKey.sort()
                    valueList = []
                    for key in unionOfKey:
                        valueList.append(item2idx[key])
                    total = vb_data[valueList[0]]
                    for k in range(1, len(valueList)):
                        total = total.__mul__(vb_data[valueList[k]])
                    support = gpuarray.sum(total).get()
                    if support >= self.minSup:
                        newBasePattern.append(" ".join(unionOfKey))
                        final[" ".join(unionOfKey)] = support

        if len(newBasePattern) > 0:
            self.eclat(newBasePattern, final, vb_data, idx2item, item2idx)

    def startMine(self):
        """
                    Frequent pattern mining process will start from here
        """
        startTime = time.time()
        basePattern = []
        final = {}

        data, lineNo = self.read_data(self.filePath, self.sep)
        vb_data, idx2item = self.compute_vertical_bitvector_data(data)
        if self.minSup < 1:
            self.minSup = int(lineNo * self.minSup)

        for i in range(len(vb_data)):
            if gpuarray.sum(vb_data[i]).get() >= self.minSup:
                basePattern.append(idx2item[i])
                final[idx2item[i]] = gpuarray.sum(vb_data[i]).get()

        # reverse idx2item
        item2idx = {idx2item[i]: i for i in idx2item}
        self.eclat(basePattern, final, vb_data, idx2item, item2idx)
        self.__time = time.time() - startTime
        self.__memRSS = psutil.Process(os.getpid()).memory_info().rss
        self.__memUSS = psutil.Process(os.getpid()).memory_full_info().uss
        self.Patterns = final
        self.__GPU_MEM = vb_data.nbytes


if __name__ == "__main__":
    filePath = "datasets\\transactional_T10I4D100K.csv"
    # filePath = "file.txt"
    sep = "\t"
    support = 500
    cudaEclatGCT = cudaEclatGCT(filePath, sep, support)
    cudaEclatGCT.startMine()
    print("Time: ", cudaEclatGCT.get_time())
    print("Memory RSS: ", cudaEclatGCT.get_memRSS())
    print("Memory USS: ", cudaEclatGCT.get_memUSS())
    print("GPU MEM: ", cudaEclatGCT.get_GPU_MEM())
    print("Patterns: ", cudaEclatGCT.get_numberOfPatterns())
