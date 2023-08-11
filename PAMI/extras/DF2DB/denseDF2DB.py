import pandas as pd
import operator

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
            Creation data base output to this outputFile.
    """

    def __init__(self, inputDF, condition, thresholdValue):
        self.inputDF = inputDF
        self.condition = condition
        self.thresholdValue = thresholdValue
        self.tids = []
        self.items = []
        self.outputFile = ' '
        self.items = list(self.inputDF.columns.values)
        self.tids = list(self.inputDF.index)


    def createTransactional(self, outputFile):
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




    def createTemporal(self, outputFile):
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
            
    def createMultipleTimeSeries(self, interval, outputFile):
        """
         :Description: Create the multiple time series data base.

         :param outputFile:  str :
                     Write multiple time series data base into outputFile

        """
        self.outputFile = outputFile
        writer = open(self.outputFile, 'w+')
        #ith open(self.outputFile, 'w+') as f:
        count = 0
        tids = []
        items= []
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
        
    def createUtility(self, outputFile):
        """
         :Description: Create the utility data base.

         :param outputFile:  str :
                     Write utility data base into outputFile

        """

        self.outputFile = outputFile
        with open(self.outputFile, 'w') as f:
            for tid in self.tids:
                #df = self.inputDF.loc[tid].dropna()
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
        :return: outputFile name
        """

        return self.outputFile


if __name__ == '__main__':
    obj = denseDF2DB(sys.argv[1], sys.argv[2], sys.argv[3])
    obj.createDB(sys.argv[4])
    transactionalDB = obj.getFileName()
    print(transactionalDB)
