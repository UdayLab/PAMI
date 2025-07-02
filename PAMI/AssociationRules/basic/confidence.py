# This code uses "confidence" metric to extract the association rules from given frequent patterns.
#
# **Importing this algorithm into a python program**
#
#             import PAMI.AssociationRules.basic import confidence as alg
#
#             iFile = 'sampleDB.txt'
#
#             minConf = 0.5
#
#             obj = alg.confidence(iFile, minConf)
#
#             obj.mine()
#
#             associationRules = obj.getAssociationRules()
#
#             print("Total number of Association Rules:", len(associationRules))
#
#             obj.save(oFile)
#
#             Df = obj.getAssociationRulesAsDataFrame()
#
#             memUSS = obj.getMemoryUSS()
#
#             print("Total Memory in USS:", memUSS)
#
#             memRSS = obj.getMemoryRSS()
#
#             print("Total Memory in RSS", memRSS)
#
#             run = obj.getRuntime()
#
#             print("Total ExecutionTime in seconds:", run)
#



__copyright__ = """
Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
     Copyright (C)  2021 Rage Uday Kiran
     
"""

import sys

from deprecated import deprecated

from PAMI.AssociationRules.basic import abstract as _ab

sys.setrecursionlimit(10**4)
from itertools import combinations
import time, psutil, os, validators, pandas as pd, urllib.request as urlopen   # whatever you aliased as _ab.*


class confidence:
    """
    About this algorithm
    ====================

    :**Description**: Association Rules are derived from frequent patterns using "confidence" metric.

    :**Reference**:

    :**Parameters**:    - **iFile** (*str*) -- *Name of the Input file to mine complete set of association rules*
                        - **oFile** (*str*) -- *Name of the Output file to write association rules*
                        - **minConf** (*float*) -- *Minimum confidence to mine all the satisfying association rules. The user can specify the minConf in float between the range of 0 to 1.*
                        - **sep** (*str*) -- *This variable is used to distinguish items from one another in a transaction. The default seperator is tab space. However, the users can override their default separator.*

    :**Attributes**:    - **startTime** (*float*) -- *To record the start time of the mining process.*
                        - **endTime** (*float*) -- *To record the completion time of the mining process.*
                        - **finalPatterns** (*dict*) -- *Storing the complete set of patterns in a dictionary variable.*
                        - **memoryUSS** (*float*) -- *To store the total amount of USS memory consumed by the program.*
                        - **memoryRSS** (*float*) -- *To store the total amount of RSS memory consumed by the program.*


    Execution methods
    =================

    **Terminal command**

    .. code-block:: console

      Format:

      (.venv) $ python3 confidence.py <inputFile> <outputFile> <minConf> <sep>

      Example Usage:

      (.venv) $ python3 confidence.py sampleDB.txt patterns.txt 0.5 ' '

    .. note:: minConf can be specified in a value between 0 and 1.
    
    
    **Calling from a python program**

    .. code-block:: python

            import PAMI.AssociationRules.basic import confidence as alg

            iFile = 'sampleDB.txt'

            minConf = 0.5

            obj = alg.confidence(iFile, minConf)

            obj.mine()

            associationRules = obj.getAssociationRules()

            print("Total number of Association Rules:", len(associationRules))

            obj.save(oFile)

            Df = obj.getAssociationRulesAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


    Credits
    =======

            The complete program was written by P. Likhitha  under the supervision of Professor Rage Uday Kiran.

    """

    _minConf = float()
    _startTime = float()
    _endTime = float()
    _iFile = " "
    _oFile = " "
    _Sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _associationRules = {}

    def __init__(self, iFile, minConf, sep="\t"):
        """
        :param iFile: input file name or path
        :type iFile: str
        :param minConf: minimum confidence
        :type minConf: float
        :param sep: Delimiter of input file
        :type sep: str
        """
        self._iFile = iFile
        self._minConf = minConf
        self._frequentPatterns = {}
        self._associationRules = []
        self._sep = sep

    def _readPatterns(self):
        """
        Populate self._frequentPatterns from a dataframe, URL or local text file.
        Accepted line-format in files:  item1<sep>item2 ... : support
        """
        fp = {}  # local scratch

        # ▸ dataframe input -----------------------------------------------------
        if isinstance(self._iFile, pd.DataFrame):
            pat_col = next(c for c in self._iFile.columns if 'pattern' in c.lower())
            sup_col = next(c for c in self._iFile.columns if 'support' in c.lower())
            for pat, sup in zip(self._iFile[pat_col], self._iFile[sup_col]):
                pat = tuple(sorted(str(pat).split(self._sep)))
                fp[pat] = int(sup)

        # ▸ URL / local file input ---------------------------------------------
        else:
            fh = urlopen.urlopen(self._iFile) if validators.url(self._iFile) \
                else open(self._iFile, encoding='utf-8')
            with fh:
                for line in fh:
                    line = line.decode() if not isinstance(line, str) else line
                    items, sup = line.strip().split(':')
                    pat = tuple(sorted(x.strip() for x in items.split(self._sep) if x))
                    fp[pat] = int(sup)

        self._frequentPatterns = fp

    def mine(self):
        """
        Create association rules that satisfy minConf.
        Stores results in self._associationRules as a list of
        (antecedent, consequent, support, confidence).
        """
        self._startTime = time.time()
        self._readPatterns()

        for itemset, sup in self._frequentPatterns.items():
            k = len(itemset)
            if k < 2:           # singleton → no rule possible
                continue

            # all non-empty proper subsets are candidate antecedents
            for r in range(1, k):
                for antecedent in combinations(itemset, r):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))

                    conf = sup / self._frequentPatterns[antecedent]
                    if conf >= self._minConf:
                        self._associationRules.append(
                            (antecedent, consequent, sup, conf)
                        )

        # bookkeeping
        self._endTime   = time.time()
        proc            = psutil.Process(os.getpid())
        self._memoryUSS = proc.memory_full_info().uss
        self._memoryRSS = proc.memory_info().rss
        print("Association rules successfully generated")


    def getMemoryUSS(self):
        """
        Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """
        Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process
        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """
        Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process
        :rtype: float
        """

        return self._endTime - self._startTime

    def getAssociationRulesAsDataFrame(self):
        """
        Return a DataFrame with **four** columns:
            Antecedent | Consequent | Support | Confidence

        :return: DataFrame containing association rules
        :rtype: pandas.DataFrame
        """
        rows = [
            {
                "Antecedent": self._sep.join(ante),
                "Consequent": self._sep.join(cons),
                "Support":    sup,
                "Confidence": conf,
            }
            for ante, cons, sup, conf in self._associationRules
        ]
        return pd.DataFrame(rows)

    def save(self, outFile:str):
        """
        Write rules in the new text format:

            itemA<sep>itemB -> itemC<sep>itemD : support : confidence
        """
        with open(outFile, 'w', encoding='utf-8') as f:
            f.write("Antecedent -> Consequent : Support : Confidence\n")
            for ante, cons, sup, conf in self._associationRules:
                lhs = self._sep.join(ante)
                rhs = self._sep.join(cons)
                f.write(f"{lhs} -> {rhs} : {sup} : {conf:.6f}\n")

    def getAssociationRules(self):
        """
        Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns
        :rtype: dict
        """
        return self._associationRules

    def printResults(self):
        """
        Function to send the result after completion of the mining process
        """
        print("Total number of Association Rules:", len(self.getAssociationRules()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())


if __name__ == "__main__":
    _ap = str()
    if len(_ab._sys.argv) == 4 or len(_ab._sys.argv) == 5:
        if len(_ab._sys.argv) == 5:
            _ap = confidence(_ab._sys.argv[1], float(_ab._sys.argv[3]), _ab._sys.argv[4])
        if len(_ab._sys.argv) == 4:
            _ap = confidence(_ab._sys.argv[1], float(_ab._sys.argv[3]))
        _ap.mine()
        _ap.mine()
        print("Total number of Association Rules:", len(_ap.getAssociationRules()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
