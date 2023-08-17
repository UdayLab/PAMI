# denseDF2DB in this code the dense dataframe is converting databases into different transactional, temporal, utility types.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.frequentPattern.basic import FPGrowth as fp
#
#     obj = fp.denseDF2DB(idf, ">=", 16)
#
#     obj.save(oFile)
#
#     Df = obj.getPatternInDataFrame()
#
#     memUSS = obj.getMemoryUSS()
#
#     print("Total Memory in USS:", memUSS)
#
#     memRSS = obj.getMemoryRSS()
#
#     print("Total Memory in RSS", memRSS)
#
#     run = obj.getRuntime()
#
#     print("Total ExecutionTime in seconds:", run)
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

import pandas as pd
import operator
import sys
from typing import List, Union

condition_operator = {
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne
}

class denseDF2DB:
    """
        :Description: This class create Data Base from DataFrame.

        :param inputDF: dataframe :
            It is dense DataFrame
        :param condition: str :
            It is condition to judge the value in dataframe
        :param thresholdValue: int or float :
            User defined value.
        :param tids: list :
            It is tids list.
        :param items: list :
            Store the items list
        :param outputFile: str  :
            Creation database output to this outputFile.


        :Attributes:

        startTime : float
          To record the start time of the mining process

        endTime : float
          To record the completion time of the mining process

        memoryUSS : float
          To store the total amount of USS memory consumed by the program

        memoryRSS : float
          To store the total amount of RSS memory consumed by the program



        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

                    from PAMI.frequentPattern.basic import FPGrowth as fp

                    obj = fp.denseDF2DB(iDdf, ">=", 16 )

                    memUSS = obj.getMemoryUSS()

                    print("Total Memory in USS:", memUSS)

                    memRSS = obj.getMemoryRSS()

                    print("Total Memory in RSS", memRSS)

                    run = obj.getRuntime()

                    print("Total ExecutionTime in seconds:", run)




        """


    def __init__(self, inputDF, condition: str, thresholdValue: Union[int, float]) -> None:
        self.inputDF = inputDF
        self.condition = condition
        self.thresholdValue = thresholdValue
        self.tids = []
        self.items = []
        self.outputFile = ' '
        self.items = list(self.inputDF.columns.values)
        self.tids = list(self.inputDF.index)


    def createTransactional(self, outputFile: str) -> None:
        """
         :Description: Create transactional data base

         :param outputFile: str :
              Write transactional data base into outputFile

        """

        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
             if self.condition not in condition_operator:
                print('Condition error')
             else:
                for tid in self.tids:
                    transaction = [item for item in self.items if condition_operator[self.condition](self.inputDF.at[tid, item], self.thresholdValue)]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f'\t{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction[0]}')
                    else:
                        continue
                    f.write('\n')




    def createTemporal(self, outputFile: str) -> None:
        """
         :Description: Create temporal data base

         :param outputFile: str :
                 Write temporal data base into outputFile

        """

        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            if self.condition not in condition_operator:
                print('Condition error')
            else:
                for tid in self.tids:
                    transaction = [item for item in self.items if condition_operator[self.condition](self.inputDF.at[tid, item], self.thresholdValue)]
                    if len(transaction) > 1:
                        f.write(f'{tid}')
                        for item in transaction:
                            f.write(f'\t{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid}')
                        f.write(f'\t{transaction[0]}')
                    else:
                        continue
                    f.write('\n')
            
    def createMultipleTimeSeries(self, interval: int, outputFile: str) -> None:
        """
         :Description: Create the multiple time series data base.

         :param outputFile:  str :
                     Write multiple time series data base into outputFile

        """
        self.outputFile = outputFile
        writer = open(self.outputFile, 'w+')
        # with open(self.outputFile, 'w+') as f:
        count = 0
        tids = []
        items = []
        values = []
        for tid in self.tids:
            count += 1
            transaction = [item for item in self.items if condition_operator[self.condition](self.inputDF.at[tid, item], self.thresholdValue)]
            for i in transaction:
                tids.append(count)
                items.append(i)
                values.append(self.inputDF.at[tid, i])
            if count == interval:
                if len(values) > 0:
                    s1, s, ss = str(), str(), str()
                    for j in range(len(tids)):
                        s1 = s1 + str(tids[j]) + '\t'
                    for j in range(len(items)):
                        s = s + items[j] + '\t'
                    for j in range(len(values)):
                        ss = ss + str(values[j]) + '\t'
                
                s2 = s1 + ':' + s + ':' + ss
                writer.write("%s\n" %s2)
                tids, items, values = [], [], []
                count = 0
        
    def createUtility(self, outputFile: str) -> None:
        """
         :Description: Create the utility database.

         :param outputFile:  str :
                     Write utility database into outputFile

        """

        self.outputFile = outputFile
        with open(self.outputFile, 'w') as f:
            for tid in self.tids:
                df = self.inputDF.loc[tid].dropna()
                f.write(f'{df.index[0]}')
                for item in df.index[1:]:
                    f.write(f'\t{item}')
                f.write(f':{df.sum()}:')
                f.write(f'{df.at[df.index[0]]}')
                for item in df.index[1:]:
                    f.write(f'\t{df.at[item]}')
                f.write('\n')

    def getFileName(self) -> str:
        """
        :return: outputFile name
        """

        return self.outputFile


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python script.py input_file condition threshold output_file")
    else:
        input_file = sys.argv[1]
        condition = sys.argv[2]
        threshold = float(sys.argv[3])  # Convert to float
        output_file = sys.argv[4]

        try:
            inputDF = pd.read_csv(input_file, index_col=0)  # Assuming input is a CSV file
            obj = denseDF2DB(inputDF, condition, threshold)
            obj.createTransactional(output_file)
            transactionalDB = obj.getFileName()
            print(f"Transactional database created and saved in '{transactionalDB}'")
        except FileNotFoundError:
            print("Error: Input file not found.")
        except ValueError:
            print("Error: Invalid threshold value. Please provide a valid number.")
