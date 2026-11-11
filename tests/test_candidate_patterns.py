"""Candidate retrieval must include rejected patterns and preserve mining results."""

import ast
import importlib
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from PAMI._candidatePatterns import CandidatePatternsMixin


TRANSACTIONS = [('A', 'B'), ('A', 'B'), ('A', 'C'), ('B', 'C'), ('D',)]
MINERS = [
    ('frequentPattern.basic.Apriori', 'Apriori', {'memorySaver': True}),
    ('frequentPattern.basic.Apriori', 'Apriori', {'memorySaver': False}),
    ('frequentPattern.basic.ECLAT', 'ECLAT', {'memorySaver': True}),
    ('frequentPattern.basic.ECLAT', 'ECLAT', {'memorySaver': False}),
    ('frequentPattern.basic.ECLATDiffset', 'ECLATDiffset', {}),
    ('frequentPattern.basic.Aprioribitset', 'Aprioribitset', {}),
    ('frequentPattern.basic.ECLATbitset', 'ECLATbitset', {}),
    ('frequentPattern.binary.BinaryApriori', 'BinaryApriori', {}),
    ('frequentPattern.binary.BinaryECLAT', 'BinaryECLAT', {}),
]


def miner_class(module, name):
    return getattr(importlib.import_module('PAMI.' + module), name)


def normalize(pattern):
    return frozenset(pattern.split('\t') if isinstance(pattern, str) else pattern)


@pytest.mark.parametrize('module,name,options', MINERS)
def test_candidates_include_rejected_itemsets_and_reset(tmp_path, module, name, options):
    cls = miner_class(module, name)
    source = tmp_path / 'transactions.txt'
    source.write_text(''.join('\t'.join(row) + '\n' for row in TRANSACTIONS))
    # Binary miners also support the standard transaction DataFrame.
    data = pd.DataFrame({'Transactions': ['\t'.join(row) for row in TRANSACTIONS]})
    obj = cls(data if '.binary.' in module else str(source), 2)
    other = cls(data if '.binary.' in module else str(source), 2)
    assert obj.getCandidatePatterns() == []
    obj.mine(**options)

    vocabulary = sorted(set().union(*map(set, TRANSACTIONS)))
    expected = {}
    for size in range(1, len(vocabulary) + 1):
        for pattern in combinations(vocabulary, size):
            support = sum(set(pattern).issubset(row) for row in TRANSACTIONS)
            if support >= 2:
                expected[frozenset(pattern)] = support
    assert {normalize(key): value for key, value in obj.getPatterns().items()} == expected

    candidates = obj.getCandidatePatterns()
    expected_candidates = {frozenset([item]) for item in vocabulary}
    expected_candidates.update(frozenset(pair) for pair in combinations('ABC', 2))
    assert {normalize(pattern) for pattern in candidates} == expected_candidates
    assert len(candidates) == 7
    assert frozenset('AC') not in expected
    assert other.getCandidatePatterns() == []

    candidates.clear()
    assert len(obj.getCandidatePatterns()) == 7
    obj.mine(**options)
    assert len(obj.getCandidatePatterns()) == 7
    source.write_text('Z\nZ\n')
    if '.binary.' in module:
        obj._iFile = pd.DataFrame({'Transactions': ['Z', 'Z']})
    obj.mine(**options)
    assert obj.getCandidatePatterns() == [('Z',)]


def test_pattern_growth_has_no_explicit_candidates(tmp_path):
    source = tmp_path / 'transactions.txt'
    source.write_text('A\tB\nA\tB\n')
    obj = miner_class('frequentPattern.basic.FPGrowth', 'FPGrowth')(str(source), 2)
    obj.mine()
    assert len(obj.getPatterns()) == 3
    assert obj.getCandidatePatterns() == []


def test_fuzzy_join_keeps_rejected_candidates(tmp_path):
    source = tmp_path / 'fuzzy.txt'
    source.write_text('A.Low\tB.Low:0.9\t0.1\nA.Low\tB.Low:0.1\t0.9\n')
    obj = miner_class('fuzzyFrequentPattern.basic.MFFIMiner', 'MFFIMiner')(str(source), 1)
    obj.mine()
    assert set(obj.getPatterns()) == {('A.Low',), ('B.Low',)}
    assert ('A.Low', 'B.Low') in obj.getCandidatePatterns()
    assert len(obj.getCandidatePatterns()) == 3


def test_utility_candidates_keep_original_labels(tmp_path):
    source = tmp_path / 'utility.txt'
    source.write_text('A\tB:10:1\t9\nA\tB:10:1\t9\nA\tC:12:1\t11\nB\tC:20:9\t11\n')
    obj = miner_class('highUtilityPattern.basic.EFIM', 'EFIM')(str(source), 15)
    obj.mine()
    candidates = {normalize(pattern) for pattern in obj.getCandidatePatterns()}
    final = {normalize(pattern) for pattern in obj.getPatterns()}
    assert frozenset('A') in candidates - final
    assert final.issubset(candidates)
    assert len(candidates) == obj._candidateCount


def test_rejected_association_rules_are_candidates(tmp_path):
    source = tmp_path / 'patterns.txt'
    source.write_text('A:4\nB:3\nA\tB:2\n')
    obj = miner_class('AssociationRules.basic.confidence', 'confidence')(str(source), 0.6)
    obj.mine()
    assert len(obj.getAssociationRules()) == 1
    assert set(obj.getCandidatePatterns()) == {(('A',), ('B',)), (('B',), ('A',))}


def test_periodic_candidates_include_failed_periodic_support(tmp_path):
    source = tmp_path / 'temporal.txt'
    source.write_text('1\tA\tB\n2\tA\tB\n3\tA\tC\n4\tB\tC\n5\tD\n')
    cls = miner_class('partialPeriodicPattern.basic.PPP_ECLAT', 'PPP_ECLAT')
    obj = cls(str(source), 1, 2)
    obj.mine()
    candidates = {normalize(pattern) for pattern in obj.getCandidatePatterns()}
    final = {normalize(pattern) for pattern in obj.getPatterns()}
    assert frozenset('AC') in candidates - final
    assert frozenset('D') in candidates - final
    assert len(candidates) == 7


def test_sequential_candidates_include_rejected_sequences(tmp_path):
    source = tmp_path / 'sequences.txt'
    source.write_text('1\t2\t-1\t3\t-1\t-2\n1\t-1\t2\t3\t-1\t-2\n1\t2\t-1\t2\t-1\t-2\n')
    obj = miner_class('sequentialPattern.basic.GSP', 'GSP')(str(source), 2)
    obj.mine()
    rejected = ('1', '-1', '1')
    assert rejected in obj.getCandidatePatterns()
    assert rejected not in obj.getPatterns()


def test_candidate_snapshots_detach_search_buffers():
    obj = CandidatePatternsMixin()
    pattern = [['A'], ['B']]
    obj._recordCandidate(pattern)
    obj._recordCandidate(pattern)
    pattern[1].append('C')
    assert obj.getCandidatePatterns() == [(('A',), ('B',))]
    obj._resetCandidatePatterns()
    assert obj.getCandidatePatterns() == []


def test_all_public_miners_expose_candidate_api():
    root = Path(__file__).resolve().parents[1] / 'PAMI'
    getters = {'getPatterns', 'getAssociationRules', 'getFrequentSubgraphs', 'getKSubgraphs'}
    checked = 0
    for path in root.rglob('*.py'):
        if 'extras' in path.parts or 'abstract' in path.name.lower():
            continue
        tree = ast.parse(path.read_text())
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            methods = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
            if not (methods & getters or (path.name == 'muse.py' and cls.name == 'Muse')):
                continue
            assert any(isinstance(base, ast.Name) and base.id == 'CandidatePatternsMixin'
                       for base in cls.bases), (path, cls.name)
            checked += 1
    assert checked >= 165
