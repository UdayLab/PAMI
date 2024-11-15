# csv2Parquet converts the input CSV file to a data frame, which is then transformed into a Parquet file.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.extras.convert import csvParquet as cp
#
#             obj = cp.CSV2Parquet(sampleDB.csv, output.parquet, sep)
#
#             obj.convert()
#
#             obj.printStats()
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
import sys
import pandas as pd
import os
import psutil
import time


class CSV2Parquet:
    """
        **About this algorithm**

        :**Description**:  This class is to convert CSV files into Parquet format.

        :**Reference**:

        :**Parameters**:    - **inputFile** (*str*) -- *Path to the input CSV file.*
                            - **outputFile** (*str*) -- *Path to the output Parquet file.*
                            - **sep** (*str*) -- *This variable is used to distinguish items from one another. The default seperator is tab space. However, the users can override their default separator.*

        :**Attributes**:    - **getMemoryUSS** (*float*) -- *Returns the memory used by the process in USS.*
                            - **getMemoryRSS** (*float*) -- *Returns the memory used by the process in RSS.*
                            - **getRuntime()** (*float*) -- *Returns the time taken to execute the conversion.*
                            - **printStats()** -- *Prints statistics about memory usage and runtime.*

        :**Methods**:       - **convert()** -- *Reads the input file, converts it to a Parquet file, and tracks memory usage and runtime.*


        **Execution methods**

        **Terminal command**

        .. code-block:: console

          Format:

          (.venv) $ python3 CSV2Parquet.py <inputFile> <outputFile> <sep>

          Example Usage:

          (.venv) $ python3 CSV2Parquet.py sampleDB.csv output.parquet \t


        **Calling from a python program**

        .. code-block:: python

                import PAMI.extras.convert.CSV2Parquet as cp

                inputFile = 'sampleDB.csv'

                sep = "\t"

                outputFile = 'output.parquet'

                obj = cp.CSV2Parquet(inputFile, outputFile, sep)

                obj.convert()

                obj.printStats()


        **Credits**

        The complete program was written by P. Likhitha  and revised by Tarun Sreepada under the supervision of Professor Rage Uday Kiran.

    """
    def __init__(self, inputFile, outputFile, sep):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.sep = sep

    def convert(self):
        """
        This function converts the input CSV file to a data frame, which is then transformed into a Parquet file.
        """
        self.start = time.time()
        file = []
        maxLen = 0
        with open(self.inputFile, "r") as f:
            for line in f:
                file.append(line.strip().split(self.sep))
                maxLen = max(maxLen, len(file[-1]))

        for i in range(len(file)):
            file[i] += [""] * (maxLen - len(file[i]))

        df = pd.DataFrame(file)

        df.to_parquet(self.outputFile)

        self.end = time.time()

        self.pid = os.getpid()
        process = psutil.Process(self.pid)
        self.memoryUSS = process.memory_full_info().uss
        self.memoryRSS = process.memory_info().rss

    def getMemoryUSS(self):
        """
        Returns the memory used by the process in USS (Unique Set Size).

        :return: The amount of memory (in bytes) used exclusively by the process
        :rtype: int
        """
        return self.memoryUSS

    def getMemoryRSS(self):
        """
        Returns the memory used by the process in RSS (Resident Set Size).

        :return: The total memory (in bytes) used by the process in RAM.
        :rtype: int
        """
        return self.memoryRSS

    def getRuntime(self):
        """
        Returns the time taken to complete the CSV to Parquet conversion.

        :return: The runtime of the conversion process in seconds.
        :rtype: float
        """
        return self.end - self.start

    def printStats(self):
        """
        Prints the resource usage statistics including memory consumption (USS and RSS) and the runtime.

        :return: Prints memory usage and runtime to the console.
        """
        print("Memory usage (USS):", self.memoryUSS)
        print("Memory usage (RSS):", self.memoryRSS)
        print("Runtime:", self.end - self.start)


if __name__ == '__main__':
    #file = "Transactional_T10I4D100K.csv"
    #sep = "\t"
    #outputFile = "output.parquet"
    if len(sys.argv) == 4:
        obj = CSV2Parquet(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError("Invalid number of arguments. Args: <inputFile> <outputFile> <separator>")
    #obj.convert()
    #obj.printStats()