"""Shared storage for explicitly generated candidate patterns."""


def _freeze(pattern):
    """Detach a candidate from mutable search buffers without losing item order."""
    if isinstance(pattern, dict):
        return frozenset((_freeze(key), _freeze(value)) for key, value in pattern.items())
    if isinstance(pattern, (list, tuple)):
        return tuple(_freeze(item) for item in pattern)
    if isinstance(pattern, (set, frozenset)):
        return frozenset(_freeze(item) for item in pattern)
    return pattern


class CandidatePatternsMixin:
    """Keep the distinct explicit candidates from the most recent mining run."""

    def _resetCandidatePatterns(self):
        self._candidatePatterns = {}

    def _recordCandidate(self, pattern):
        if '_candidatePatterns' not in self.__dict__:
            self._resetCandidatePatterns()
        self._candidatePatterns.setdefault(_freeze(pattern), None)

    def _recordCandidates(self, patterns):
        for pattern in patterns:
            self._recordCandidate(pattern)

    def getCandidatePatterns(self):
        """Return distinct explicit candidates, including rejected candidates.

        The returned list is a snapshot of the latest run. Candidates use
        immutable representations (tuples for itemsets and nested sequences).
        Returns an empty list before mining and for algorithms without explicit
        candidate generation. Use ``len(obj.getCandidatePatterns())`` to count
        candidates; this is not the number of support tests or search nodes.

        :return: candidate patterns from the latest mining run
        :rtype: list
        """
        return list(getattr(self, '_candidatePatterns', {}))
