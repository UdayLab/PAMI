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

import os
import mmap
import time
import psutil
from joblib import Parallel, delayed

class efim:
    """
    EFIM is one of the fastest algorithm to mine High Utility ItemSets from transactional databases.
    
    Reference:
    ---------
        Zida, S., Fournier-Viger, P., Lin, J.CW. et al. EFIM: a fast and memory efficient algorithm for
        high-utility itemset mining. Knowl Inf Syst 51, 595–625 (2017). https://doi.org/10.1007/s10115-016-0986-0
    
    Attributes:
    ----------
        inputFile (str): The input file path.
        minUtil (int): The minimum utility threshold.
        sep (str): The separator used in the input file.
        threads (int): The number of threads to use.
        Patterns (dict): A dictionary containing the discovered patterns.
        rename (dict): A dictionary containing the mapping between the item IDs and their names.
        runtime (float): The runtime of the algorithm in seconds.
        memoryRSS (int): The Resident Set Size (RSS) memory usage of the algorithm in bytes.
        memoryUSS (int): The Unique Set Size (USS) memory usage of the algorithm in bytes.

    Methods:
    -------
        read_file(): Read the input file and return the filtered transactions, primary items, and secondary items.
        binarySearch(arr, item): Perform a binary search on the given array to find the given item.
        project(beta, file_data, secondary): Project the given beta itemset on the given database.
        search(collections): Search for high utility itemsets in the given collections.
        startMine(): Start the EFIM algorithm.
        savePatterns(outputFile): Save the patterns discovered by the algorithm to an output file.
        getPatterns(): Get the patterns discovered by the algorithm.
        getRuntime(): Get the runtime of the algorithm.
        getMemoryRSS(): Get the Resident Set Size (RSS) memory usage of the algorithm.
        getMemoryUSS(): Get the Unique Set Size (USS) memory usage of the algorithm.
        printResults(): Print the results of the algorithm.

    """

    def __init__(self, inputFile, minUtil, sep = '\t', threads = 1):
        self.inputFile = inputFile
        self.minUtil = minUtil
        self.sep = sep
        self.Patterns = {}
        self.rename = {}
        self.threads = threads

    # Read input file
    def read_file(self):
        file_data = []
        twu = {}

        with open(self.inputFile, 'r') as f:
            fd = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            for line in iter(fd.readline, b""):
                line = line.decode('utf-8').strip().split(":")
                
                # Parse and process the line
                line = [x.split(self.sep) for x in line]
                weight = int(line[1][0])

                # Update file data with the parsed items
                file_data.append([line[0], [int(x) for x in line[2]]])

                for k in line[0]:
                    if k not in twu:
                        twu[k] = weight
                    else:
                        twu[k] += weight

        # Filter TWU dictionary based on minUtil (minimum utility threshold)
        twu = {k: v for k, v in twu.items() if v >= self.minUtil}

        # Sort TWU items by utility
        twu = {k: v for k, v in sorted(twu.items(), key=lambda item: item[1], reverse=True)}

        strToInt = {}
        t = len(twu)
        for k in twu.keys():
            strToInt[k] = t
            self.rename[t] = k
            t -= 1

        secondary = set(self.rename.keys())

        # Filter and sort transactions
        subtree = {}
        filtered_transactions = {}
        for col in file_data:
            zipped = zip(col[0], col[1])
            transaction = [(strToInt[x], y) for x, y in zipped if x in strToInt]
            transaction = sorted(transaction, key=lambda x: x[0])
            if len(transaction) > 0:
                val = [x[1] for x in transaction]
                key = [x[0] for x in transaction]
                
                fs = frozenset(key)

                if fs not in filtered_transactions:
                    filtered_transactions[fs] = [key, val, 0]
                else:
                    filtered_transactions[fs][1] = [x + y for x, y in zip(filtered_transactions[fs][1], val)]

                subUtil = sum([x[1] for x in transaction])
                temp = 0

                for i in range(len(transaction)):
                    item = key[i]
                    if item not in subtree:
                        subtree[item] = subUtil - temp
                    else:
                        subtree[item] += subUtil - temp
                    temp += val[i]

        primary = [key for key in subtree.keys() if subtree[key] >= self.minUtil]

        return filtered_transactions, primary, secondary
    

    def binarySearch(self, arr, item):
        low = 0
        high = len(arr) - 1
        mid = 0

        while low <= high:
            mid = (high + low) // 2
            if arr[mid] < item:
                low = mid + 1
            elif arr[mid] > item:
                high = mid - 1
            else:
                return mid

        return -1

    def project(self, beta, file_data, secondary):
        projected_db = {}
        local_utils = {}
        subtree_utils = {}
        utility = 0     

        added = set()

        item = beta[-1]

        temp = [v for k, v in file_data.items() if item in k]
        start = time.time()

        for v in temp:
            index = self.binarySearch(v[0], item)

            curr = v[1][index] + v[2]
            utility += curr

            newKey = []
            newVal = []

            for i in range(index+1, len(v[0])):
                if v[0][i] in secondary:
                    newKey.append(v[0][i])
                    newVal.append(v[1][i])

            if len(newKey) == 0:
                continue

            s = sum(newVal)
            temp = 0

            for i in range(len(newKey)):
                if newKey[i] in added:
                    local_utils[newKey[i]] += s + curr
                    subtree_utils[newKey[i]] += s + curr - temp
                else:
                    local_utils[newKey[i]] = s + curr
                    subtree_utils[newKey[i]] = s + curr - temp
                    added.add(newKey[i])
                
                temp += newVal[i]
            
            fs = frozenset(newKey)

            if fs not in projected_db:
                projected_db[fs] = [newKey, newVal, curr]
            else:
                projected_db[fs][1] = [x + y for x, y in zip(projected_db[fs][1], newVal)]
                projected_db[fs][2] += curr

        nprimary = [key for key in subtree_utils.keys() if subtree_utils[key] >= self.minUtil]
        nsecondary = set([key for key in local_utils.keys() if local_utils[key] >= self.minUtil])

        return beta, projected_db, nsecondary, nprimary, utility
    

    def search(self, collections):

        if (self.threads > 1):
            with Parallel(n_jobs=self.threads) as parallel:
                while len(collections) > 0:
                    print("Collections:", len(collections), "Patterns:", len(self.Patterns))
                    new_collections = []

                    # print("Num of tasks:", sum(len(collections[i][2]) for i in range(len(collections))))
                    results = parallel(delayed(self.project)(collections[i][0] + [collections[i][2][j]], collections[i][1], collections[i][3]) for i in range(len(collections)) for j in range(len(collections[i][2])))

                    for i in range(len(results)):
                        beta, projected_db, secondary, primary, utility = results[i]
                        if utility >= self.minUtil:
                            self.Patterns[tuple(beta)] = utility
                        if len(primary) > 0:
                            new_collections.append([beta, projected_db, primary, secondary])
                    
                    collections = new_collections

        else:
            while len(collections) > 0:
                print("Collections:", len(collections), "Patterns:", len(self.Patterns))
                new_collections = []
                for i in range(len(collections)):
                    for j in range(len(collections[i][2])):
                        beta, projected_db, secondary, primary, utility = self.project(collections[i][0] + [collections[i][2][j]], collections[i][1], collections[i][3])
                        if utility >= self.minUtil:
                            self.Patterns[tuple(beta)] = utility
                        if len(primary) > 0:
                            new_collections.append([beta, projected_db, primary, secondary])

                collections = new_collections



    def startMine(self):
        """
        Start the EFIM algorithm.

        Returns:
            None
        """

        ps = psutil.Process(os.getpid())

        self.start = time.time()

        fileData, primary, secondary = self.read_file()
        print("time to read file: " + str(time.time() - self.start))

        collection = [[[], fileData, primary, secondary]]

        self.search(collection)

        self.memoryRSS = ps.memory_info().rss
        self.memoryUSS = ps.memory_full_info().uss

        end = time.time()
        self.runtime = end - self.start

    def savePatterns(self, outputFile):
        """
        Save the patterns discovered by the algorithm to an output file.

        Args:
            outputFile (str): The output file path.

        Returns:
            None
        """

        with open(outputFile, 'w') as f:
            for key, value in self.Patterns.items():
                key = [self.rename[x] for x in key]
                joined = " ".join(key) + " #UTIL: " + str(value) + "\n"
                f.write(joined)

    def getPatterns(self):
        """
        Get the patterns discovered by the algorithm.

        Returns:
            dict: A dictionary containing the discovered patterns.
        """
        return self.Patterns

    def getRuntime(self):
        """
        Get the runtime of the algorithm.

        Returns:
            float: The runtime in seconds.
        """
        return self.runtime

    def getMemoryRSS(self):
        """
        Get the Resident Set Size (RSS) memory usage of the algorithm.

        Returns:
            int: The RSS memory usage in bytes.
        """
        return self.memoryRSS

    def getMemoryUSS(self):
        """
        Get the Unique Set Size (USS) memory usage of the algorithm.

        Returns:
            int: The USS memory usage in bytes.
        """
        return self.memoryUSS


    def printResults(self):
        print("Total number of High Utility Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in seconds:", self.getRuntime())


if __name__ == "__main__":

    inputFile = 'EFIM/accidents_utility_spmf.txt'
    minUtil = 28000000

    # inputFile = 'EFIM/chainstore.txt'
    # minUtil = 2500000

    # inputFile = 'test.txt'
    # minUtil = 5

    # inputFile = "EFIM/BMS_utility_spmf.txt"
    # minUtil = 2025000

    # inputFile = "EFIM/Utility_pumsb.csv"
    # minUtil = 7500000

    sep = " "
    f = efim(inputFile, minUtil, sep, 1)
    f.startMine()
    print("# of patterns: " + str(len(f.getPatterns())))
    print("Time taken: " + str(f.getRuntime()))
    f.savePatterns("mine.txt")

