from PAMI.extras.DF2DB.denseDF2DB import *
from PAMI.extras.DF2DB.sparseDF2DB import *

class DF2DB:
    """
        :Description: This class create database from DataFrame. Threshold values and conditions are defined to all item.


        :param inputDF: DataFrame :
             It is sparse or dense DataFrame
        :param thresholdValue: int or float :
             It is threshold value of all item
        :param condition: str :
             It is condition of all item
        :param DFtype: str :
             It is DataFrame type. It should be sparse or dense. Default DF is sparse.


        """
    def __init__(self, inputDF, thresholdValue, condition, DFtype='sparse'):
        self.inputDF = inputDF
        self.thresholdValue = thresholdValue
        self.condition = condition
        self.DFtype = DFtype.lower()
        if DFtype == 'sparse':
            self.DF2DB = sparseDF2DB(self.inputDF, self.condition, self.thresholdValue)
        elif DFtype == 'dense':
            self.DF2DB = denseDF2DB(self.inputDF, self.condition, self.thresholdValue)
        else:
            raise Exception('DF type should be sparse or dense')

    def getTransactional(self, outputFile):
        """
        create transactional database and return outputFileName

        :param outputFile: file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createTransactional(outputFile)
        return self.DF2DB.getFileName()

    def getTemporal(self, outputFile):
        """
        create temporal database and return outputFile name

        :param outputFile: file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createTemporal(outputFile)
        return self.DF2DB.getFileName()

    def getUtility(self, outputFile):
        """
        create utility database and return outputFile name

        :param outputFile:  file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createUtility(outputFile)
        return self.DF2DB.getFileName()
