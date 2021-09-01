from PAMI.DF2DB.denseDF2DB import *
from PAMI.DF2DB.sparseDF2DB import *

class DF2DB:
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
