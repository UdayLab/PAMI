from PAMI.DF2DB.denseDF2DBPlus import *
from PAMI.DF2DB.sparseDF2DBPlus import *


class DF2DBPlus:
    def __init__(self, inputDF, thresholdConditionDF, DFtype='sparse'):
        self.inputDF = inputDF
        self.thresholdConditionDF = thresholdConditionDF
        self.DFtype = DFtype.lower()
        if DFtype == 'sparse':
            self.DF2DB = sparseDF2DBPlus(self.inputDF, self.thresholdConditionDF)
        elif DFtype == 'dense':
            self.DF2DB = denseDF2DBPlus(self.inputDF, self.thresholdConditionDF)
        else:
            print('type error')

    def getDB(self, outputFile):
        self.DF2DB.createDB(outputFile)
        return self.DF2DB.getFileName()

    def getTDB(self, outputFile):
        self.DF2DB.createTDB(outputFile)
        return self.DF2DB.getFileName()

    def getUDB(self, outputFile):
        self.DF2DB.createUDB(outputFile)
        return self.DF2DB.getFileName()
