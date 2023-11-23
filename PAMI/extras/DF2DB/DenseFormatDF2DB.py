# DenseFormatDF2DB in this code the dense dataframe is converting databases into different transactional, temporal, utility types.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
#     from PAMI.extras.DF2DB import DenseFormatDF2DB as db
#
#     obj = db.DenseFormatDF2DB(idf, ">=", 16)
#
#     obj.save(oFile)
#
#     obj.convert2TransactionalDatabase("outputFileName") # To create transactional database
#
#     obj.convert2TemporalDatabase("outputFileName") # To create temporal database
#
#     obj.convert2MultipleTimeSeries("outputFileName") # To create Mutliple TimeSeries database
#
#     obj.convert2UtilityDatabase("outputFileName") # To create utility database
#
#     obj.getFileName("outputFileName") # To get file name of the database

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

import operator
from typing import Union

condition_operator = {
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne
}


class DenseFormatDF2DB:
    """
        :Description: This class create Data Base from DataFrame.

        :param inputDF: dataframe :
            It is dense DataFrame
        :param condition: str :
            It is condition to judge the value in dataframe
        :param thresholdValue: int or float :
            User defined value.


        **Importing this algorithm into a python program**
        --------------------------------------------------------
        .. code-block:: python

        from PAMI.extras.DF2DB import DenseFormatDF2DB as db

        obj = db.DenseFormatDF2DB(iDdf, ">=", 16 )

        obj.convert2TransactionalDatabase("outputFileName") # To create transactional database

        obj.convert2TemporalDatabase("outputFileName") # To create temporal database

        obj.convert2MultipleTimeSeries("outputFileName") # To create Multiple TimeSeries database

        obj.convert2UtilityDatabase("outputFileName") # To create utility database

        obj.getFileName("outputFileName") # To get file name of the database


    """

    def __init__(self, inputDF) -> None:
        self.inputDF = inputDF
        self.tids = []
        self.items = []
        self.outputFile = ' '
        self.items = list(self.inputDF.columns.values)
        self.tids = list(self.inputDF.index)

    def convert2TransactionalDatabase(self, outputFile: str, condition: str, thresholdValue: Union[int, float]) -> None:
        """
         :Description: Create transactional data base

         :param outputFile: str :
              Write transactional data base into outputFile

         :param condition: str :
            It is condition to judge the value in dataframe
         :param thresholdValue: int or float :
            User defined value.

        """


        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            if condition not in condition_operator:
                print('Condition error')
            else:
                for tid in self.tids:
                    transaction = [item for item in self.items if
                                   condition_operator[condition](self.inputDF.at[tid, item], thresholdValue)]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f'\t{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction[0]}')
                    else:
                        continue
                    f.write('\n')

    def convert2TemporalDatabase(self, outputFile: str, condition: str, thresholdValue: Union[int, float]) -> None:
        """
         :Description: Create temporal data base

         :param outputFile: str :
                 Write temporal data base into outputFile
         :param condition: str :
            It is condition to judge the value in dataframe
         :param thresholdValue: int or float :
            User defined value.

        """

        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            if condition not in condition_operator:
                print('Condition error')
            else:
                for tid in self.tids:
                    transaction = [item for item in self.items if
                                   condition_operator[condition](self.inputDF.at[tid, item], thresholdValue)]
                    if len(transaction) > 1:
                        f.write(f'{tid + 1}')
                        for item in transaction:
                            f.write(f'\t{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid + 1}')
                        f.write(f'\t{transaction[0]}')
                    else:
                        continue
                    f.write('\n')

    def convert2MultipleTimeSeries(self, interval: int, outputFile: str, condition: str,
                                   thresholdValue: Union[int, float]) -> None:
        """
         :Description: Create the multiple time series database.

         :param outputFile:  str :
                     Write multiple time series database into outputFile.

        :param interval: int:
                    Breaks the given timeseries into intervals.
        :param condition: str :
            It is condition to judge the value in dataframe
        :param thresholdValue: int or float :
            User defined value.

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
            transaction = [item for item in self.items if
                           condition_operator[condition](self.inputDF.at[tid, item], thresholdValue)]
            for i in transaction:
                tids.append(count)
                items.append(i)
                values.append(self.inputDF.at[tid, i])
            if count == interval:
                s1, s, ss = str(), str(), str()
                if len(values) > 0:

                    for j in range(len(tids)):
                        s1 = s1 + str(tids[j]) + '\t'
                    for j in range(len(items)):
                        s = s + items[j] + '\t'
                    for j in range(len(values)):
                        ss = ss + str(values[j]) + '\t'

                s2 = s1 + ':' + s + ':' + ss
                writer.write("%s\n" % s2)
                tids, items, values = [], [], []
                count = 0

    def convert2UncertainTransactional(self, outputFile: str, condition: str,
                                       thresholdValue: Union[int, float]) -> None:
        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            if condition not in condition_operator:
                print('Condition error')
            else:
                for tid in self.tids:
                    transaction = [item for item in self.items if
                                   condition_operator[condition](self.inputDF.at[tid, item], thresholdValue)]
                    uncertain = [self.inputDF.at[tid, item] for item in self.items if
                                 condition_operator[condition](self.inputDF.at[tid, item], thresholdValue)]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f'\t{item}')
                        f.write(f':')
                        for value in uncertain:
                            tt = 0.1 + 0.036 * abs(25 - value)
                            tt = round(tt, 2)
                            f.write(f'\t{tt}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction[0]}')
                        tt = 0.1 + 0.036 * abs(25 - uncertain[0])
                        tt = round(tt, 2)
                        f.write(f':{tt}')
                    else:
                        continue
                    f.write('\n')

    def convert2UtilityDatabase(self, outputFile: str) -> None:
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

# Dataframes do not run from a terminal

# if __name__ == '__main__':
#     obj = DenseFormatDF2DB(sys.argv[1], sys.argv[2], sys.argv[3])
#     obj.convert2TransactionalDatabase(sys.argv[4])
#     transactionalDB = obj.getFileName()
#     print(transactionalDB)