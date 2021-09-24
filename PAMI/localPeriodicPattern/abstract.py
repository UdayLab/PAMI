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
import time
import csv
import pandas as pd
from collections import defaultdict
from itertools import combinations as c
import os
import os.path
import psutil


class localPeriodicPatterns(ABC):
    """ This abstract base class defines the variables and methods that every frequent pattern mining algorithm must
        employ in PAMI


        Attributes:
        ----------
        inputFile : str
            Input file name or path of the input file
        minSup: integer or float or str
            The user can specify minSup either in count or proportion of database size.
            If the program detects the data type of minSup is integer, then it treats minSup is expressed in count.
            Otherwise, it will be treated as float.
            Example: minSup=10 will be treated as integer, while minSup=10.0 will be treated as float
        startTime:float
            To record the start time of the algorithm
        endTime:float
            To record the completion time of the algorithm
        finalPatterns: dict
            Storing the complete set of patterns in a dictionary variable
        outputFile : str
            Name of the output file to store complete set of frequent patterns
        memoryUSS : float
            To store the total amount of USS memory consumed by the program
        memoryRSS : float
            To store the total amount of RSS memory consumed by the program

        Methods:
        -------
        startMine()
            Mining process will start from here
        getLocalPeriodicPatterns()
            Complete set of patterns will be retrieved with this function
        savePatterns(oFile)
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

    def __init__(self, iFile, maxPer, maxSoPer, minDur):
        """

        :param iFile: Input file name or path of the input file
        :type iFile: str
        :param maxPer: The user can specify maxPer either in count or proportion of database size.
            If the program detects the data type of maxPer is integer, then it treats maxPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxPer=10 will be treated as integer, while maxPer=10.0 will be treated as float
        :type maxPer: int or float or str
        :param maxSoPer: The user can specify maxSoPer either in count or proportion of database size.
            If the program detects the data type of maxSoPer is integer, then it treats maxSoPer is expressed in count.
            Otherwise, it will be treated as float.
            Example: maxSoPer=10 will be treated as integer, while maxSoPer=10.0 will be treated as float.
        :type maxSoPer: int or float or str
        :param minDur: The user can specify minDur either in count or proportion of database size.
            If the program detects the data type of minDur is integer, then it treats minDur is expressed in count.
            Otherwise, it will be treated as float.
            Example: minDur=10 will be treated as integer, while minDur=10.0 will be treated as float.
        :type minDur: int or float or str
        """

        self.iFile = iFile
        self.maxPer = maxPer
        self.maxSoPer = maxSoPer
        self.minDur = minDur

    @abstractmethod
    def iFile(self):
        """Variable to store the input file path/file name"""

        pass

    @abstractmethod
    def maxPer(self):
        """Variable to store the user-specified minimum support value"""

        pass

    @abstractmethod
    def maxSoPer(self):
        """Variable to store the user-specified minimum support value"""

        pass

    @abstractmethod
    def minDur(self):
        """Variable to store the user-specified minimum support value"""

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
        """Variable to store USS memory consumed by the program"""

        pass

    @abstractmethod
    def memoryRSS(self):
        """Variable to store RSS memory consumed by the program"""

        pass

    @abstractmethod
    def finalPatterns(self):
        """Variable to store the complete set of patterns in a dictionary"""

        pass

    @abstractmethod
    def oFile(self):
        """Variable to store the name of the output file to store the complete set of local periodic patterns"""

        pass

    @abstractmethod
    def startMine(self):
        """Code for the mining process will start from this function"""

        pass

    @abstractmethod
    def getLocalPeriodicPatterns(self):
        """Complete set of local periodic patterns generated will be retrieved from this function"""

        pass

    @abstractmethod
    def savePatterns(self, oFile):
        """Complete set of local periodic patterns will be saved in to an output file from this function

        :param oFile: Name of the output file
        :type oFile: file
        """

        pass

    @abstractmethod
    def getPatternsAsDataFrame(self):
        """Complete set of local periodic patterns will be loaded in to data frame from this function"""

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
