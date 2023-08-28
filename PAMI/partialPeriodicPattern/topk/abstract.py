#  Copyright (C)  2021 Rage Uday Kiran
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
import time as _time
import math as _math
import csv as _csv
import pandas as _pd
from collections import defaultdict as _defaultdict
from itertools import combinations as _combinations
import os as _os
import os.path as _path
import psutil as _psutil
import sys as _sys
import validators as _validators
from urllib.request import urlopen as _urlopen


class partialPeriodicPatterns(ABC):
    """ This abstract base class defines the variables and methods that every periodic-frequent pattern mining algorithm must
        employ in PAMI

    Attributes:
    ----------
        iFile : str
            Input file name or path of the input file
        k: int or float or str
            The user can specify minPS either in count or proportion of database size.
            If the program detects the data type of minPS is integer, then it treats minPS is expressed in count.
            Otherwise, it will be treated as float.
            Example: minPS=10 will be treated as integer, while minPS=10.0 will be treated as float
        period: int or float or str
            The user can specify period either in count or proportion of database size.
            If the program detects the data type of period is integer, then it treats period is expressed in count.
            Otherwise, it will be treated as float.
            Example: period=10 will be treated as integer, while period=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        startTime:float
            To record the start time of the algorithm
        endTime:float
            To record the completion time of the algorithm
        finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        oFile : str
            Name of the output file to store complete set of periodic-frequent patterns
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program

        Methods
        -------
        startMine()
            Mining process will start from here
        getPatterns()
            Complete set of patterns will be retrieved with this function
        save(oFile)
            Complete set of periodic-frequent patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of periodic-frequent patterns will be loaded in to data frame
        getMemoryUSS()
            Total amount of USS memory consumed by the program will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the program will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the program will be retrieved from this function
    """

    def __init__(self, iFile, k, period, sep = '\t'):
        """
        :param iFile: Input file name or path of the input file
        :type iFile: str
        :param k: The user can specify minPS either in count or proportion of database size.
            If the program detects the data type of minPS is integer, then it treats k is expressed in count.
            Otherwise, it will be treated as float.
            Example: k=10 will be treated as integer, while minPS=10.0 will be treated as float
        :type k: int or float or str
        :param period: The user can specify period either in count or proportion of database size.
            If the program detects the data type of period is integer, then it treats period is expressed in count.
            Otherwise, it will be treated as float.
            Example: period=10 will be treated as integer, while period=10.0 will be treated as float
        :type period: int or float or str
        :param sep: separator used in user specified input file
        :type sep: str
        """

        self._iFile = iFile
        self._k = k
        self._period = period
        self._sep = sep
        self._oFile = str()
        self._finalPatterns = {}
        self._memoryUSS = float()
        self._memoryRSS = float()
        self._startTime = float()
        self._endTime = float()


    @abstractmethod
    def startMine(self):
        """Code for the mining process will start from this function"""

        pass

    @abstractmethod
    def getPatterns(self):
        """Complete set of periodic-frequent patterns generated will be retrieved from this function"""

        pass

    @abstractmethod
    def save(self, oFile):
        """Complete set of periodic-frequent patterns will be saved in to an output file from this function

        :param oFile: Name of the output file
        :type oFile: file
        """

        pass

    @abstractmethod
    def getPatternsAsDataFrame(self):
        """Complete set of periodic-frequent patterns will be loaded in to data frame from this function"""

        pass

    @abstractmethod
    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the program will be retrieved from this function"""

        pass

    @abstractmethod
    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the program will be retrieved from this function"""
        pass

    @abstractmethod
    def getRuntime(self):
        """Total amount of runtime taken by the program will be retrieved from this function"""

        pass

    @abstractmethod
    def printResults(self):
        """ To print all the results of execution"""

        pass
