#  Copyright (C)  2024 Rage Uday Kiran , P Krishna Reddy
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



from abc import ABC as ABC, abstractmethod as _abstractmethod
import time as _time
import validators as _validators
from urllib.request import urlopen as _urlopen
import csv as _csv
import pandas as _pd
from collections import defaultdict as _defaultdict
from itertools import combinations as _c
import os as _os
import os.path as _ospath
import psutil as _psutil
from array import *
import functools as _functools
import sys as _sys


class GTCP(ABC):
    def __init__(self,iFile,minsup,minGTC,minGTPC,maxOR=0.2):
        """
            iFile : input file
            minsup : Minimum support 
            minGTC : Minimum Graph transaction coverage
            minGTPC : Minimum graph pattern coverage 
            maxOR : Maximum overlap ratio
        """

    @_abstractmethod
    def mine(self):
        """
            Mine the coverage patterns
        """
    
    @_abstractmethod
    def getPatterns(self):
        """
            Get all the coverage patterns
        """

    @_abstractmethod    
    def Coverage(self,g):
        """
            Get coverage of a graph
            param
                g : Graph id
        """

    @_abstractmethod
    def patternCoverage(self,pattern):
        """
            Get patternCoverage of a pattern
            param
                pattern: pattern for which pattern coverage needs to be computed
        """

    @_abstractmethod
    def OverlapRatio(self,pattern):
        """
            Get Overlap ratio of a pattern
            param
                pattern: pattern for which overlap ratio needs to be computed
        """


    @_abstractmethod
    def GetFIDBasedFlatTransactions(self):
        """
            Convert into FID based transactions
        """

    @_abstractmethod
    def getallFreq1(self):
        """
            Get all the Patterns of size 1

        """

    @_abstractmethod
    def join(self,l1,l2):
        """
            Join two patterns
            Param 
                l1: Pattern 1
                l2: Pattern 2
        """

    @_abstractmethod
    def Cmine(self):
       """
        Cmine Algorithm for mining coverage patterns

       """






