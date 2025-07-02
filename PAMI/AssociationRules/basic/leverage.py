# This code uses "leverage" metric to extract the association rules from given frequent patterns.
#
# **Importing this algorithm into a python program**
#
#             import PAMI.AssociationRules.basic import leverage as alg
#
#             obj = alg.leverage(iFile, minLev)
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

from PAMI.AssociationRules.basic import abstract as _ab
from deprecated import deprecated
# increase reucursion depth
import os
import sys
sys.setrecursionlimit(10**4)
from itertools import combinations

from itertools import combinations
import os, time, psutil, pandas as pd, validators, urllib.request as urlopen


class leverage:
    """
    About this algorithm
    ====================

    :**Description**: Association Rules are derived from frequent patterns using "leverage" metric.

    :**Reference**:

    :**Parameters**:    - **iFile** (*str*) -- *Name of the Input file to mine complete set of association rules*
                        - **oFile** (*str*) -- *Name of the Output file to write association rules*
                        - **minLev** (*float*) -- *Minimum leverage to mine all the satisfying association rules. The user can specify the minLev in float between the range of 0 to 1.*
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

      (.venv) $ python3 leverage.py <inputFile> <outputFile> <minLev> <sep>

      Example Usage:

      (.venv) $ python3 leverage.py sampleDB.txt patterns.txt 0.5 ' '

    .. note:: minLev can be specified in a value between 0 and 1.
    
    
    **Calling from a python program**

    .. code-block:: python

            import PAMI.AssociationRules.basic import leverage as alg

            obj = alg.leverage(iFile, minLev)

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

    _minLev = float()
    _startTime = float()
    _endTime = float()
    _iFile = " "
    _oFile = " "
    _Sep = " "
    _memoryUSS = float()
    _memoryRSS = float()
    _associationRules = {}

    def __init__(self,
             iFile,  # frequent patterns  (DataFrame | path | URL)
             minLev: float,
            sep: str = "\t",
            dbLen: int | None = None,
            dbFile: str | None = None):
        """
        Parameters
        ----------
        iFile   : frequent-pattern source (DataFrame, file, or URL)
        minLev  : minimum leverage threshold
        sep     : item delimiter in text files (default '\\t')
        dbLen   : total #transactions *if you already know it*
        dbFile  : path / URL of the original transaction DB.
                  Used **only** when supports are absolute counts
                  and dbLen was not supplied.
        """
        self._iFile = iFile
        self._minLev = minLev
        self._sep = sep
        self._dbLen = dbLen
        self._dbFile = dbFile

        self._frequentPatterns = {}
        self._associationRules = []
        self._startTime = self._endTime = 0.0
        self._memoryUSS = self._memoryRSS = 0.0

    def _count_db_lines(self) -> int:
        """Return #lines in the transaction DB (minus blank lines)."""
        if not self._dbFile:
            return 0
        fh = urlopen.urlopen(self._dbFile) if validators.url(self._dbFile) \
            else open(self._dbFile, encoding='utf-8')
        with fh:
            return sum(1 for ln in fh if ln.strip())

    def _readPatterns(self):
        """Load frequent patterns into `_frequentPatterns`."""
        fp = {}

        # ▲ DataFrame source ----------------------------------------------------
        if isinstance(self._iFile, pd.DataFrame):
            pat_col = next(c for c in self._iFile.columns if 'pattern' in c.lower())
            sup_col = next(c for c in self._iFile.columns if 'support' in c.lower())
            for pat, sup in zip(self._iFile[pat_col], self._iFile[sup_col]):
                pat = tuple(sorted(str(pat).split(self._sep)))
                fp[pat] = float(sup)

        # ▲ URL or local file ---------------------------------------------------
        else:
            fh = urlopen.urlopen(self._iFile) if validators.url(self._iFile) \
                 else open(self._iFile, encoding='utf-8')
            with fh:
                for line in fh:
                    line = line.decode() if not isinstance(line, str) else line
                    items, sup = [s.strip() for s in line.strip().split(':', 1)]
                    pat = tuple(sorted(it for it in items.split(self._sep) if it))
                    fp[pat] = float(sup)

        # sort patterns by length and then lexicographically

        if any(v > 1 for v in fp.values()):
            denom = (self._dbLen  # user-supplied
                     or self._count_db_lines()  # NEW  ←
                     or max(fp.values()))  # fallback guess
            fp = {k: v / denom for k, v in fp.items()}

        self._frequentPatterns = fp


    def mine(self):
        """Generate rules with leverage ≥ `minLev`."""
        self._startTime = time.time()
        self._readPatterns()

        for itemset, sup_xy in self._frequentPatterns.items():
            k = len(itemset)
            if k < 2:
                continue
            for r in range(1, k):
                for ante in combinations(itemset, r):
                    ante = tuple(sorted(ante))
                    cons = tuple(sorted(set(itemset) - set(ante)))
                    lev = sup_xy - self._frequentPatterns[ante] * self._frequentPatterns[cons]
                    if lev >= self._minLev:
                        self._associationRules.append((ante, cons, sup_xy, lev))

        self._endTime = time.time()
        proc = psutil.Process(os.getpid())
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

    def getAssociationRules(self):
        return self._associationRules

    def getAssociationRulesAsDataFrame(self):
        rows = [
            {"Antecedent": self._sep.join(a),
             "Consequent": self._sep.join(c),
             "Support":    s,
             "Leverage":   l}
            for a, c, s, l in self._associationRules
        ]
        return pd.DataFrame(rows)

    def save(self, outFile: str):
        with open(outFile, 'w', encoding='utf-8') as f:
            f.write(f"Antecedent{self._sep}Consequent{self._sep}Support{self._sep}Leverage\n")
            for a, c, s, l in self._associationRules:
                f.write(f"{self._sep.join(a)} -> {self._sep.join(c)} : {s} : {l}\n")


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
    if len(_ab._sys.argv) == 5 or len(_ab._sys.argv) == 6:
        if len(_ab._sys.argv) == 6:
            _ap = leverage(_ab._sys.argv[1], float(_ab._sys.argv[3]), int(_ab._sys.argv[4]),_ab._sys.argv[5])
        if len(_ab._sys.argv) == 5:
            _ap = leverage(_ab._sys.argv[1],float(_ab._sys.argv[3]),int(_ab.sys.argv[4]),sep='\t')
        _ap.mine()
        _ap.mine()
        print("Total number of Association Rules:", len(_ap.getAssociationRules()))
        _ap.save(_ab._sys.argv[2])
        print("Total Memory in USS:", _ap.getMemoryUSS())
        print("Total Memory in RSS", _ap.getMemoryRSS())
        print("Total ExecutionTime in ms:", _ap.getRuntime())
    else:
        print("Error! The number of input parameters do not match the total number of parameters provided")
