import pandas as pd

class denseDF2DB:
    """
        This class create Data Base from DataFrame.

        Attribute:
        ----------
        inputDF : pandas.DataFrame
            It is dense DataFrame
        condition : str
            It is condition to judge the value in dataframe
        thresholdValue : int or float:
            User defined value.
        tids : list
            It is tids list.
        items : list
            Store the items list
        outputFile : str
            Creation data base output to this outputFile.

        Methods:
        --------
        createDB(outputFile)
            Create transactional data base from dataFrame
        createTDB(outputFile)
            Create temporal dataBase from dataFrame
        createUDB(outputFile)
            Create utility database from dataFrame
        getFileName()
            Return outputFileName.
        """

    def __init__(self, inputDF, condition, thresholdValue):
        self.inputDF = inputDF
        self.condition = condition
        self.thresholdValue = thresholdValue
        self.tids = []
        self.items = []
        self.outputFile = ' '
        self.items = list(self.inputDF.columns.values)[1:]
        self.inputDF = self.inputDF.set_index('tid')
        self.tids = list(self.inputDF.index)


    def createTransactional(self, outputFile):
        """
        Create transactional data base

        :param outputFile: Write transactional data base into outputFile
        :type outputFile: str
        """

        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            if self.condition == '>':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] > self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction}')
                    else:
                        continue
                    f.write('\n')

            elif self.condition == '>=':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] >= self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction}')
                    else:
                        continue
                    f.write('\n')

            elif self.condition == '<=':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] <= self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction}')
                    else:
                        continue
                    f.write('\n')

            elif self.condition == '<':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] < self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction}')
                    else:
                        continue
                    f.write('\n')
            elif self.condition == '==':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] == self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction}')
                    else:
                        continue
                    f.write('\n')
            elif self.condition == '!=':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] != self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{transaction[0]}')
                        for item in transaction[1:]:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{transaction}')
                    else:
                        continue
                    f.write('\n')
            else:
                print('Condition error')



    def createTemporal(self, outputFile):
        """
        Create temporal data base

        :param outputFile: Write temporal data base into outputFile
        :type outputFile: str
        """

        self.outputFile = outputFile
        with open(outputFile, 'w') as f:
            if self.condition == '>':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] > self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{tid}')
                        for item in transaction:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid}')
                        f.write(f',{transaction}')
                    else:
                        continue
                    f.write('\n')

            elif self.condition == '>=':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] >= self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{tid}')
                        for item in transaction:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid}')
                        f.write(f',{transaction}')
                    else:
                        continue
                    f.write('\n')

            elif self.condition == '<=':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] <= self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{tid}')
                        for item in transaction:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid}')
                        f.write(f',{transaction}')
                    else:
                        continue
                    f.write('\n')

            elif self.condition == '<':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] < self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{tid}')
                        for item in transaction:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid}')
                        f.write(f',{transaction}')
                    else:
                        continue
                    f.write('\n')
            elif self.condition == '==':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] == self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{tid}')
                        for item in transaction:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid}')
                        f.write(f',{transaction}')
                    else:
                        continue
                    f.write('\n')
            elif self.condition == '!=':
                for tid in self.tids:
                    transaction = [item for item in self.items if self.inputDF.at[tid, item] != self.thresholdValue]
                    if len(transaction) > 1:
                        f.write(f'{tid}')
                        for item in transaction:
                            f.write(f',{item}')
                    elif len(transaction) == 1:
                        f.write(f'{tid}')
                        f.write(f',{transaction}')
                    else:
                        continue
                    f.write('\n')

            else:
                print('Condition error')

    def createUtility(self, outputFile):
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

    def getFileName(self):
        """
        return outputFile name

        :return: outputFile name
        """

        return self.outputFile


if __name__ == '__main__':
    DF = createDenseDF('denseDF.csv')
    obj = denseDF2DB(DF.getDF(), '>=', 2)
    obj.createDB('testTransactional.csv')
    transactionalDB = obj.getFileName()
    print(transactionalDB)
