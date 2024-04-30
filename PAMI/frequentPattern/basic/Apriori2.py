from typing import Dict, List, Set, Union
from datetime import datetime
from deprecated import deprecated
import pandas as pd
import psutil
import os

class Apriori:
    """
    Apriori algorithm for frequent pattern mining in transactional databases.

    Args:
        iFile (str): Input file name or path of the input file.
        minSup (Union[int, float, str]): Minimum support threshold. If int, treated as count. If float, treated as proportion of database size.
        sep (str, optional): Separator used to distinguish items from each other in a transaction. Default is '\t'.

    Attributes:
        minSup (float): Minimum support threshold.
        startTime (float): Start time of the mining process.
        endTime (float): End time of the mining process.
        frequentPatterns (Dict[str, int]): Dictionary storing the complete set of patterns.
        database (List[Set[str]]): List to store transactions of the database.

    Methods:
        mine(): Perform the frequent pattern mining process.
        getMemoryUsage(): Get the total memory consumed.
        getRuntime(): Get the total runtime of the mining process.
        getPatternsAsDataFrame(): Get frequent patterns as a DataFrame.
        savePatterns(outFile): Save the final patterns into a file.
        getPatterns(): Get the set of frequent patterns.
        printResults(): Print the results of the execution.
    """

    def __init__(self, iFile: str, minSup: Union[int, float, str], sep: str = '\t'):
        self.minSup = self._convertMinSup(minSup)
        self.startTime = 0.0
        self.endTime = 0.0
        self.frequentPatterns = {}
        self.database = self._loadDatabase(iFile, sep)

    def _convertMinSup(self, minSup: Union[int, float, str]) -> float:
        if isinstance(minSup, int):
            return minSup
        elif isinstance(minSup, float):
            return len(self.database) * minSup
        elif isinstance(minSup, str):
            if '.' in minSup:
                return len(self.database) * float(minSup)
            else:
                return int(minSup)

    def _loadDatabase(self, iFile: str, sep: str) -> List[Set[str]]:
        database = []
        with open(iFile, 'r') as f:
            for line in f:
                items = line.strip().split(sep)
                database.append(set(items))
        return database

    def mine(self) -> None:
        """
        Perform the frequent pattern mining process.
        """
        self.startTime = datetime.now()
        candidates = [{item} for transaction in self.database for item in transaction]
        frequentSets = []
        while candidates:
            counts = self._countCandidates(candidates)
            frequentSets.extend([c for c in candidates if counts[tuple(c)] >= self.minSup])
            candidates = self._generateCandidates(frequentSets)
        self.frequentPatterns = {self._setToStr(pattern): self._getSupport(pattern) for pattern in frequentSets}
        self.endTime = datetime.now()

    def _countCandidates(self, candidates: List[Set[str]]) -> Dict[tuple, int]:
        counts = {}
        for transaction in self.database:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    counts[tuple(candidate)] = counts.get(tuple(candidate), 0) + 1
        return counts

    def _generateCandidates(self, frequentSets: List[Set[str]]) -> List[Set[str]]:
        newCandidates = []
        for i, pattern1 in enumerate(frequentSets):
            for pattern2 in frequentSets[i + 1:]:
                if list(pattern1)[:-1] == list(pattern2)[:-1]:
                    newCandidate = pattern1.union(pattern2)
                    if all(self._isSubset(subset, frequentSets) for subset in self._getSubsets(newCandidate)):
                        newCandidates.append(newCandidate)
        return newCandidates

    def _isSubset(self, subset: Set[str], superset: List[Set[str]]) -> bool:
        return any(subset.issubset(pattern) for pattern in superset)

    def _getSubsets(self, pattern: Set[str]) -> List[Set[str]]:
        return [set(subset) for subset in self._powerSet(pattern) if subset]

    def _powerSet(self, pattern: Set[str]) -> List[List[str]]:
        return [list(subset) for i in range(len(pattern) + 1) for subset in combinations(pattern, i)]

    def _setToStr(self, pattern: Set[str]) -> str:
        return '\t'.join(sorted(pattern))

    def _getSupport(self, pattern: Set[str]) -> int:
        return sum(pattern.issubset(transaction) for transaction in self.database)

    def getMemoryUsage(self) -> float:
        """
        Get the total memory consumed.

        Returns:
            float: Total memory consumed.
        """
        process = psutil.Process(os.getpid())
        return process.memory_full_info().uss

    def getRuntime(self) -> float:
        """
        Get the total runtime of the mining process.

        Returns:
            float: Total runtime in seconds.
        """
        return (self.endTime - self.startTime).total_seconds()

    def getPatternsAsDataFrame(self) -> pd.DataFrame:
        """
        Get frequent patterns as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing frequent patterns.
        """
        data = [[pattern, support] for pattern, support in self.frequentPatterns.items()]
        return pd.DataFrame(data, columns=['Patterns', 'Support'])

    def savePatterns(self, outFile:
