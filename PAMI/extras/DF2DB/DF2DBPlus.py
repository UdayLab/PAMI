from PAMI.extras.DF2DB.denseDF2DBPlus import *
from PAMI.extras.DF2DB.sparseDF2DBPlus import *


class DF2DBPlus:
    """
            :Description:  This class create database from DataFrame. Threshold values and conditions are defined to each item by thresholdConditonDF.

            :param inputDF: DataFrame :
                 It is sparse or dense DataFrame
            :param thresholdConditionDF: pandas.DataFrame :
                It is DataFrame to contain threshold values and condition each item
            :param condition: str :
                 It is condition of all item
            :param DFtype: str :
                 It is DataFrame type. It should be sparse or dense. Default DF is sparse.


    """


    def __init__(self, inputDF, thresholdConditionDF, DFtype='sparse'):
        self.inputDF = inputDF
        self.thresholdConditionDF = thresholdConditionDF
        self.DFtype = DFtype.lower()
        if DFtype == 'sparse':
            self.DF2DB = sparseDF2DBPlus(self.inputDF, self.thresholdConditionDF)
        elif DFtype == 'dense':
            self.DF2DB = denseDF2DBPlus(self.inputDF, self.thresholdConditionDF)
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

    def getTDB(self, outputFile):
        """
        create temporal database and return outputFile name

        :param outputFile: file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createTemporal(outputFile)
        return self.DF2DB.getFileName()

    def getUDB(self, outputFile):
        """
        create utility database and return outputFile name

        :param outputFile:  file name or path to store database
        :type outputFile: str

        :return: outputFile name
        """
        self.DF2DB.createUtility(outputFile)
        return self.DF2DB.getFileName()
