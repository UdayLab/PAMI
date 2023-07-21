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

from abc import ABC as _ABC, abstractmethod as _abstractmethod
import time as _time
import math as _math
import csv as _csv
import pandas as _pd
from collections import defaultdict as _defaultdict
from itertools import combinations as _c
import os as _os
import os.path as _ospath
import psutil as _psutil
import sys as _sys
import validators as _validators
from urllib.request import urlopen as _urlopen


class _coveragePatterns(_ABC):
    """ This abstract base class defines the variables and methods that every coverage pattern mining algorithm must
        employ in PAMI

       Attributes
        ----------
        iFile : str
            Input file name or path of the input file
        minCS: int or float or str
            The user can specify minCS either in count or proportion of database size.
            If the program detects the data type of minCS is integer, then it treats minCS is expressed in count.
            Otherwise, it will be treated as float.
            Example: minCS=10 will be treated as integer, while minCS=10.0 will be treated as float
        maxOR: int or float or str
            The user can specify maxOR either in count or proportion of database size.
            If the program detects the data type of maxOR is integer, then it treats maxOR is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxOR=10 will be treated as integer, while maxOR=10.0 will be treated as float
        minRF: int or float or str
            The user can specify minRF either in count or proportion of database size.
            If the program detects the data type of minRF is integer, then it treats minRF is expressed in count.
            Otherwise, it will be treated as float.
            Example: minRF=10 will be treated as integer, while minRF=10.0 will be treated as float
        sep : str
            This variable is used to distinguish items from one another in a transaction. The default seperator is tab space or \t.
            However, the users can override their default separator.
        startTime: float
            To record the start time of the algorithm
        endTime: float
            To record the completion time of the algorithm
        finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        oFile : str
            Name of the output file to store complete set of coverage patterns
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
            Complete set of coverage patterns will be loaded in to a output file
        getPatternsAsDataFrame()
            Complete set of coverage patterns will be loaded in to data frame
        getMemoryUSS()
            Total amount of USS memory consumed by the program will be retrieved from this function
        getMemoryRSS()
            Total amount of RSS memory consumed by the program will be retrieved from this function
        getRuntime()
            Total amount of runtime taken by the program will be retrieved from this function
    """

    def __init__(self, iFile, minRF, minCS, maxOR, sep='\t'):
        """
        :param iFile: Input file name or path of the input file
        :type iFile: str
        :param minRF: The user can specify minimum relative frequency either in count or proportion of database size.
            If the program detects the data type of minRF is integer, then it treats minRF is expressed in count.
            Otherwise, it will be treated as float.
            Example: minRF=10 will be treated as integer, while minRF=10.0 will be treated as float
        :type minRF: int or float or str
        :param minCS: The user can specify minimum coverage support either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        :param maxOR: The user can specify maximum overlap ratio either in count or proportion of database size.
        :type maxOR: int or float or str
        :param sep: separator used in user specified input file
        :type sep: str
        """

        self._iFile = iFile
        self._minCS = minCS
        self._minRF = minRF
        self._maxOR = maxOR
        self._sep = sep
        self._finalPatterns = {}
        self._startTime = float()
        self._endTime = float()
        self._memoryRSS = float()
        self._memoryUSS = float()
        self._oFile = " "

    @_abstractmethod
    def startMine(self):
        """Code for the mining process will start from this function"""

        pass

    @_abstractmethod
    def getPatterns(self):
        """Complete set of coverage patterns generated will be retrieved from this function"""

        pass

    @_abstractmethod
    def save(self, oFile):
        """Complete set of coverage patterns will be saved in to an output file from this function

        :param oFile: Name of the output file
        :type oFile: file
        """

        pass

    @_abstractmethod
    def getPatternsAsDataFrame(self):
        """Complete set of coverage patterns will be loaded in to data frame from this function"""

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
        """ To print the results of the execution"""

        pass