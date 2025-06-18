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

#

# import pandas as pd
# def fasta_to_csv(filename):
#   f=open(filename,"r+")
#   st=""
#   ind=0
#   for i in f.readlines():
#       if ind != 0:
#           i=i.replace("\n","")
#           st+=i
#       ind+=1
#   df=pd.DataFrame([st])
#   df.columns=["seq"]
#   f.close()
#   return df



import pandas as pd
import time
import psutil
import argparse
import os

class Fasta2CSV:
    """
    **About this algorithm**

    :**Description**:  This class is to convert FASTA format into CSV file.

    :**Parameters**:    - **inputFile** (*str*) -- *Path to the input FASTA file.*
                         - **outputFile** (*str*) -- *Path to the output CSV file.*

    :**Methods**:       - **convert()** -- *Reads the FASTA file, converts it to a CSV file.*

    **Execution methods**

    **Terminal command**

    .. code-block:: console

      Format:
        python fasta_to_csv.py <input.fasta> <output.csv>
    """

    def __init__(self, inputFile, outputFile):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.process = psutil.Process(os.getpid())

    def convert(self):
        df = self.fasta_to_csv(self.inputFile)
        df.to_csv(self.outputFile, index=False)

    def fasta_to_csv(self,filename):
        self.start_time = time.time()
        with open(filename, "r") as f:
            st = ""
            ind = 0
            for line in f:
                if ind != 0:
                    line = line.strip()
                    st += line
                ind += 1
        df = pd.DataFrame([st], columns=["seq"])

        self.end_time = time.time()
        return df

    def getMemoryUSS(self):
        return self.process.memory_full_info().uss

    def getMemoryRSS(self):
        return self.process.memory_info().rss

    def getRuntime(self):
        return self.end_time - self.start_time

    def printStats(self):
        print(f"Memory used (USS): {self.getMemoryUSS() / (1024 * 1024):.2f} MB")
        print(f"Memory used (RSS): {self.getMemoryRSS() / (1024 * 1024):.2f} MB")
        print(f"Runtime: {self.getRuntime():.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA file to CSV format.")
    parser.add_argument("inputFile", type=str, help="Path to the input FASTA file.")
    parser.add_argument("outputFile", type=str, help="Path to the output CSV file.")

    args = parser.parse_args()

    converter = Fasta2CSV(args.inputFile, args.outputFile)
    converter.convert()
    converter.printStats()
