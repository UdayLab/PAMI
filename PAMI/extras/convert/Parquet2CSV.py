# Parquet2CSV is a code used to converts the input Parquet file into a CSV file by the specified separator.
#
# **Importing this algorithm into a python program**
#
#             from PAMI.extras.convert import Parquet2CSV as p2c
#
#             obj = p2c.Parquet2CSV(input.parquet, sampleDB.csv, sep)
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

class Parquet2CSV:
    """
        **About this algorithm**

        :**Description**:  This class is to convert Parquet format into CSV file.

        :**Reference**:

        :**Parameters**:    - **inputFile** (*str*) -- *Path to the input Parquet file.*
                            - **outputFile** (*str*) -- *Path to the output CSV file.*
                            - **sep** (*str*) -- *This variable is used to distinguish items from one another. The default seperator is tab space. However, the users can override their default separator.*

        :**Attributes**:    - **getMemoryUSS** (*int*) -- *Returns the memory used by the process in USS.*
                            - **getMemoryRSS** (*int*) -- *Returns the memory used by the process in RSS.*
                            - **getRuntime()** (*float*) -- *Returns the time taken to execute the conversion.*
                            - **printStats()** -- * Prints statistics about memory usage and runtime.*

        :**Methods**:       - **convert()** -- *Reads the Parquet file, converts it to a CSV file, and tracks memory usage and runtime.*

        **Execution methods**

        **Terminal command**

        .. code-block:: console

          Format:

          (.venv) $ python3 _CSV2Parquet.py <inputFile> <outputFile> <sep>

          Example Usage:

          (.venv) $ python3 _CSV2Parquet.py output.parquet sampleDB.csv \t


        **Calling from a python program**

        .. code-block:: python

                import PAMI.extras.convert.Parquet2CSV as pc

                inputFile = 'output.parquet'

                sep = "\t"

                outputFile = 'sampleDB.csv'

                obj = pc.Parquet2CSV(inputFile, outputFile, sep)

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
        This function converts the input Parquet file into a CSV file where each row is joined by the specified separator and written to the output file.
        """
        self.start = time.time()
        df = pd.read_parquet(self.inputFile)

        with open(self.outputFile, "w") as f:
            for i in range(len(df)):
                f.write(self.sep.join(df.iloc[i]) + "\n")

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
        Returns the time taken to complete the Parquet to CSV conversion.

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
    #sep = "\t"
    #outputFile = "output.csv"
    if len(sys.argv) == 4:
        obj = Parquet2CSV(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError("Invalid number of arguments. Args: <inputFile> <outputFile> <separator>")
    #obj.convert()
    #obj.printStats()
