from PAMI.extras.DF2DB.denseDF2DB import *
from PAMI.extras.DF2DB.sparseDF2DB import *

class DF2DB:
    """
        This class create database from DataFrame. Threshold values and conditions are defined to all item.

        Attribute:
        ----------
        inputDF : pandas.DataFrame
            It is sparse or dense DataFrame
        thresholdValue : int or float
            It is threshold value of all item
        condition : str
            It is condition of all item
        DFtype : str
            It is DataFrame type. It should be sparse or dense. Default DF is sparse.

        Nethods:
        --------
        getDB(outputFile)
            Create transactional database from DataFrame and store into outputFile
        getTDB(outputFile)
            Create temporal database from DataFrame and store into outputFile
        getUDB(outputFile)
            Create utility database from DataFrame and store into outputFile
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
