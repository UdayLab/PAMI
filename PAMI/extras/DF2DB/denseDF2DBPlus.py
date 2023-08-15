# denseDF2DBPlus in this code the dense dataframe is converting databases into different transactional, temporal, utility types.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.frequentPattern.basic import FPGrowth as fp
#
#     obj = fp.denseDF2DBPlus(idf, ">=", 16)
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

class denseDF2DBPlus:
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

                    obj = fp.denseDF2DBPlus(iDdf, ">=", 16)

                    memUSS = obj.getMemoryUSS()

                    print("Total Memory in USS:", memUSS)

                    memRSS = obj.getMemoryRSS()

                    print("Total Memory in RSS", memRSS)

                    run = obj.getRuntime()

                    print("Total ExecutionTime in seconds:", run)





        """

    def __init__(self, inputDF, thresholdConditionDF) -> None:
        self.inputDF = inputDF.T
        self.thresholdConditionDF = thresholdConditionDF
        self.tids = []
        self.items = []
        self.outputFile = ' '
        self.items = list(self.inputDF.index)
        self.tids = list(self.inputDF.columns)
        self.df = pd.merge(self.inputDF, self.thresholdConditionDF, left_index=True, right_index=True)


    def createTransactional(self, outputFile: str) -> None:
        """
        Create transactional data base

        :param outputFile: Write transactional data base into outputFile
        :type outputFile: str
        """

        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            for tid in self.tids:
                transaction = [item for item in self.items if
                               (self.df.at[item, 'condition'] == '>' and self.df.at[item, tid] > self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '>=' and self.df.at[item, tid] >= self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '<=' and self.df.at[item, tid] <= self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '<' and self.df.at[item, tid] < self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '==' and self.df.at[item, tid] == self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '!=' and self.df.at[item, tid] != self.df.at[item, 'threshold'])]
                if len(transaction) > 1:
                    f.write(f'{transaction[0]}')
                    for item in transaction[1:]:
                        f.write(f',{item}')
                elif len(transaction) == 1:
                    f.write(f'{transaction[0]}')
                else:
                    continue
                f.write('\n')



    def createTemporal(self, outputFile: str) -> None:
        """
        Create temporal data base

        :param outputFile: Write temporal data base into outputFile
        :type outputFile: str
        """

        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            for tid in self.tids:
                transaction = [item for item in self.items if
                               (self.df.at[item, 'condition'] == '>' and self.df.at[item, tid] > self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '>=' and self.df.at[item, tid] >= self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '<=' and self.df.at[item, tid] <= self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '<' and self.df.at[item, tid] < self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '==' and self.df.at[item, tid] == self.df.at[item, 'threshold']) or
                               (self.df.at[item, 'condition'] == '!=' and self.df.at[item, tid] != self.df.at[item, 'threshold'])]
                if len(transaction) > 1:
                    f.write(f'{tid}')
                    for item in transaction:
                        f.write(f',{item}')
                elif len(transaction) == 1:
                    f.write(f'{tid}')
                    f.write(f',{transaction[0]}')
                else:
                    continue
                f.write('\n')

    def createUtility(self, outputFile: str) -> None:
        """
        Create the utility data base.

        :param outputFile: Write utility data base into outputFile
        :type outputFile: str
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
