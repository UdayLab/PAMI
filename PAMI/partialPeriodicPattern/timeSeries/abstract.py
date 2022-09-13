from abc import ABC as _ABC, abstractmethod as _abstractmethod
import time as _time
import math as _math
import csv as _csv
import pandas as _pd
from collections import defaultdict as _defaultdict
from itertools import combinations as _combinations
import os as _os
import os.path as _ospath
import psutil as _psutil
import sys as _sys
import validators as _validators
from urllib.request import urlopen as _urlopen


class _partialPeriodicPatterns(_ABC):
    """ This abstract base class defines the variables and methods that every frequent pattern mining algorithm must
    employ in PAMI
        ...
        Attributes
        ----------
        iFile : str
            Input file name or path of the input file
        minSup: float
            UserSpecified minimum support value. It has to be given in terms of count of total number of transactions
            in the input database/file
        startTime:float
            To record the start time of the algorithm
        endTime:float
            To record the completion time of the algorithm
        finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        oFile : str
            Name of the output file to store complete set of frequent patterns
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program
        Methods
        -------
        startMine()
            Mining process will start from here
        getFrequentPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of frequent patterns will be loaded in to data frame
        getMemoryUSS()
            Total amount of USS memory consumed by the program will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the program will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the program will be retrieved from this function
    """

    def __init__(self, iFile, periodicSupport, period, sep='\t'):
        """
        :param iFile: Input file name or path of the input file
        :type iFile: str
        :param minSup: UserSpecified minimum support value. It has to be given in terms of count of total number of
        transactions in the input database/file
        :type minSup: float
        """

        self._iFile = iFile
        self._periodicSupport = periodicSupport
        self._period = period
        self._sep = sep
        self._finalPatterns = {}
        self._oFile = str()
        self._startTime = float()
        self._endTime = float()
        self._memoryUSS = float()
        self._memoryRSS = float()

    '''@abstractmethod
    def iFile(self):
        """Variable to store the input file path/file name"""
        pass
    @abstractmethod
    def periodicSupport(self):
        """Variable to store the user-specified minimum support value"""
        pass
    def period(self):
        """Variable to store the user specified maximum periodicity value"""
        pass
    @abstractmethod
    def sep(self):
        """Variable to store the input file path/file name"""
        pass
    @abstractmethod
    def startTime(self):
        """Variable to store the start time of the mining process"""
        pass
    @abstractmethod
    def endTime(self):
        """Variable to store the end time of the complete program"""
        pass
    @abstractmethod
    def memoryUSS(self):
        """Variable to store the end time of the complete program"""
        pass
    @abstractmethod
    def memoryRSS(self):
        """Variable to store the end time of the complete program"""
        pass
    @abstractmethod
    def finalPatterns(self):
        """Variable to store the complete set of patterns in a dictionary"""
        pass
    @abstractmethod
    def oFile(self):
        """Variable to store the name of the output file to store the complete set of frequent patterns"""
        pass'''

    @_abstractmethod
    def startMine(self):
        """Code for the mining process will start from this function"""

        pass

    @_abstractmethod
    def getPatterns(self):
        """Complete set of frequent patterns generated will be retrieved from this function"""

        pass

    @_abstractmethod
    def save(self, oFile):
        """Complete set of frequent patterns will be saved in to an output file from this function
        :param oFile: Name of the output file
        :type oFile: file
        """

        pass

    @_abstractmethod
    def getPatternsAsDataFrame(self):
        """Complete set of frequent patterns will be loaded in to data frame from this function"""

        pass

    @_abstractmethod
    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the program will be retrieved from this function"""

        pass

    @_abstractmethod
    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the program will be retrieved from this function"""
        pass

    @_abstractmethod
    def getRuntime(self):
        """Total amount of runtime taken by the program will be retrieved from this function"""

        pass

    @_abstractmethod
    def printResults(self):
        """ To print all the results of execution"""

        pass
