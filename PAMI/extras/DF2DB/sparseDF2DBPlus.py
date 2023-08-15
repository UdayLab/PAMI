# sparseDF2DBPlus in this code the dense dataframe is converting databases into different transactional, temporal, utility types.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.frequentPattern.basic import FPGrowth as fp
#
#     obj = fp.sparseDF2DBPlus(idf, ">=", 16)
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

class sparseDF2DBPlus:
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
                Creation data base output to this outputFile.

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

                    obj = fp.sparseDF2DBPlus(iDdf, ">=", 16)

                    memUSS = obj.getMemoryUSS()

                    print("Total Memory in USS:", memUSS)

                    memRSS = obj.getMemoryRSS()

                    print("Total Memory in RSS", memRSS)

                    run = obj.getRuntime()

                    print("Total ExecutionTime in seconds:", run)





         """


    def __init__(self, inputDF, thresholdConditionDF) -> None:
        self.inputDF = inputDF
        self.thresholdConditionDF = thresholdConditionDF
        self.outputFile = ''
        self.df = pd.merge(self.inputDF, self.thresholdConditionDF, left_on='item', right_index=True)
        self.df.query('(condition == ">" & value > threshold) | (condition == ">=" & value >= threshold) |'
                      '(condition == "<=" & value <= threshold) | (condition == "<" & value < threshold) |'
                      '(condition == "==" & value == threshold) | (condition == "!=" & value != threshold)',
                      inplace=True)
        self.df = self.df.drop(columns=['value', 'threshold', 'condition'])
        self.df = self.df.groupby(level=0)['item'].apply(list)

    def createTransactional(self, outputFile: str) -> None:
        """
        Create transactional data base

        :param outputFile: Write transactional data base into outputFile
        :type outputFile: str
        """

        self.outputFile = outputFile
        with open(self.outputFile, 'w') as f:
            for line in self.df:
                f.write(f'{line[0]}')
                for item in line[1:]:
                    f.write(f',{item}')
                f.write('\n')

    def createTemporal(self, outputFile: str) -> None:
        """
        Create temporal data base

        :param outputFile: Write temporal data base into outputFile
        :type outputFile: str
        """

        self.outputFile = outputFile
        with open(self.outputFile, 'w') as f:
            for tid in self.df.index:
                f.write(f'{tid}')
                for item in self.df[tid]:
                    f.write(f',{item}')
                f.write('\n')

    def createUtility(self, outputFile: str) -> None:
        """
        Create the utility data base.

        :param outputFile: Write utility data base into outputFile
        :type outputFile: str
        """

        self.outputFile = outputFile
        items = self.inputDF.groupby(level=0)['item'].apply(list)
        values = self.inputDF.groupby(level=0)['value'].apply(list)
        sums = self.inputDF.groupby(level=0)['value'].sum()
        index = list(items.index)
        with open(self.outputFile, 'w') as f:
            for tid in index:
                f.write(f'{items[tid][0]}')
                for item in items[tid][1:]:
                    f.write(f'\t{item}')
                f.write(f':{sums[tid]}:')
                f.write(f'{values[tid][0]}')
                for value in values[tid][1:]:
                    f.write(f'\t{value}')
                f.write('\n')


    def getFileName(self) -> str:
        """


        :return: outputFile name
        """

        return self.outputFile
