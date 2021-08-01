import pandas as pd

class sparseDF2DB:
    """
    This class create Data Base from DataFrame.
    Attribute:
    ----------
    inputDF : pandas.DataFrame
        It is sparse DataFrame
    condition : str
        It is condition to judge the value in dataframe
    thresholdValue : int or float:
        User defined value.
    DF : pandas.DataFrame
        It is data frame to create data base.
    outputFile : str
        Creation data base output to this outputFile.

    Methods:
    --------
    createDB(outputFile)
        Create transactional data base from dataFrame
    createTDB(outputFile)
        Create temporal dataBase from dataFrame
    getFileName()
        Return outputFileName.
    """

    def __init__(self, inputDF, condition, thresholdValue):
        self.inputDF = inputDF
        self.condition = condition
        self.thresholdValue = thresholdValue
        self.DF = pd.DataFrame
        self.outputFile = ''
        if self.condition == '>':
            self.df = self.inputDF.query(f'value > {self.thresholdValue}')
        elif self.condition == '>=':
            self.df = self.inputDF.query(f'value >= {self.thresholdValue}')
        elif self.condition == '<=':
            self.df = self.inputDF.query(f'value <= {self.thresholdValue}')
        elif self.condition == '<':
            self.df = self.inputDF.query(f'value < {self.thresholdValue}')
        else:
            print('Condition error')
        self.df = self.df.drop(columns='value')
        self.df = self.df.groupby('tid')['item'].apply(list)


    def createDB(self, outputFile):
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

    def getFileName(self):
        """
        return outputFile name
        :return: outputFile name
        """
        return self.outputFile

    def createTDB(self, outputFile):
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


