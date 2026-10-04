MFFIMiner
=========

MFFIMiner mines multiple fuzzy frequent itemsets from a transformed fuzzy
transactional database. It retains every frequent region of an item. Each
pattern contains at most one region of the same base item, so ``milk.Low``
and ``milk.High`` can occur in separate patterns but never together.

The implementation uses sparse fuzzy lists and depth-first search.

Reference: Lin et al., `Efficient Mining of Multiple Fuzzy Frequent Itemsets
<https://doi.org/10.1007/s40815-016-0246-1>`_.

The `MFFIMiner notebook
<https://github.com/UdayLab/PAMI/blob/main/notebooks/fuzzyFrequentPattern/basic/MFFIMiner.ipynb>`_
contains the examples below and a comparison of minimum support thresholds.

Installation
------------

Use a PAMI installation that includes ``MFFIMiner``. To install a source
checkout, run the following command from the repository root:

.. code-block:: console

   python3 -m pip install -e .

This is the CPU implementation; it does not require CUDA.

Input parameters
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Parameter
     - Type
     - Description
   * - ``iFile``
     - str or pandas.DataFrame
     - Local file path, URL, or DataFrame containing transformed fuzzy data.
   * - ``minSup``
     - int, float, or str
     - Positive integer fuzzy support threshold, or a float proportion in
       ``(0, 1]``. Numeric strings follow the same rules.
   * - ``sep``
     - str
     - Non-empty separator for item labels and memberships. Default: tab
       (``'\t'``).

Input format
------------

Each text row contains item labels and their memberships separated by a colon:

.. code-block:: text

   item.Region<sep>item.Region:membership<sep>membership

Use the same separator on both sides of the colon. The memberships correspond
to the item labels in the same order. They must be finite numbers between
``0`` and ``1``, inclusive. A fuzzy item label must not appear twice in one
transaction.

The final dot separates the base item from its region. For example,
``milk.Low`` and ``milk.High`` are two regions of ``milk``. A label without
a dot is treated as its own base item.

The following sample uses spaces as the separator:

.. code-block:: text

   milk.Low milk.High bread.High:0.8 0.2 0.9
   milk.Low milk.High bread.High:0.3 0.7 0.6
   milk.Low milk.High bread.High:0.6 0.4 0.8
   milk.Low milk.High bread.High:0.1 0.9 0.4

A DataFrame must contain ``Transactions`` and ``fuzzyValues`` columns. Each
cell may contain a list, or a string using ``sep``. Extra columns are ignored.

Fuzzy support and minimum support
---------------------------------

A pattern's contribution in one transaction is the minimum membership of its
items. If an item is absent, the contribution is zero. Fuzzy support is the
sum of these contributions over all transactions.

For the sample above, the support of ``milk.High bread.High`` is
``min(0.2, 0.9) + min(0.7, 0.6) + min(0.4, 0.8) + min(0.9, 0.4) = 1.6``.
This is a fuzzy support value, rather than a count of transactions containing
both items.

``minSup=1`` uses an absolute fuzzy support threshold of ``1``.
``minSup=0.25`` uses ``0.25 * number_of_transactions``. On the four-row
sample, both thresholds are equivalent. Likewise, the string ``'1'`` means
an absolute threshold, while ``'1.0'`` means all transactions
(``1.0 * number_of_transactions``).

Patterns with support equal to the threshold are included.

Mining a local file
-------------------

Save the sample as ``fuzzyTransactions.txt``, then run:

.. code-block:: python

   from PAMI.fuzzyFrequentPattern.basic import MFFIMiner as alg

   obj = alg.MFFIMiner('fuzzyTransactions.txt', minSup=1, sep=' ')
   obj.mine()
   patterns = obj.getPatterns()
   patternsDF = obj.getPatternsAsDataFrame()
   obj.save('mffiPatterns.txt')
   obj.printResults()

Use ``mine()`` to start mining; ``startMine()`` is deprecated.

The sample produces five patterns:

.. list-table::
   :header-rows: 1

   * - Pattern
     - Fuzzy support
   * - ``bread.High``
     - 2.7
   * - ``milk.High``
     - 2.2
   * - ``milk.Low``
     - 1.8
   * - ``bread.High milk.High``
     - 1.6
   * - ``bread.High milk.Low``
     - 1.8

``getPatterns()`` returns a dictionary with tuples of sorted item labels as
keys and fuzzy supports as values. ``getPatternsAsDataFrame()`` returns
``Patterns`` and ``Support`` columns; the labels in ``Patterns`` are joined
using ``sep``.

``save()`` writes one pattern per line in the following format:

.. code-block:: text

   fuzzyItem<sep>fuzzyItem:support

Mining a DataFrame
------------------

.. code-block:: python

   import pandas as pd
   from PAMI.fuzzyFrequentPattern.basic import MFFIMiner as alg

   transactions = pd.DataFrame({
       'Transactions': [
           ['milk.Low', 'milk.High', 'bread.High'],
           ['milk.Low', 'milk.High', 'bread.High'],
           ['milk.Low', 'milk.High', 'bread.High'],
           ['milk.Low', 'milk.High', 'bread.High']
       ],
       'fuzzyValues': [
           [0.8, 0.2, 0.9],
           [0.3, 0.7, 0.6],
           [0.6, 0.4, 0.8],
           [0.1, 0.9, 0.4]
       ]
   })
   obj = alg.MFFIMiner(transactions, minSup=0.25)
   obj.mine()
   patternsDF = obj.getPatternsAsDataFrame()

Mining a URL
------------

.. code-block:: python

   from PAMI.fuzzyFrequentPattern.basic import MFFIMiner as alg

   inputURL = 'https://u-aizu.ac.jp/~udayrage/datasets/fuzzyDatabases/Fuzzy_T10I4D100K.csv'
   obj = alg.MFFIMiner(inputURL, minSup=0.05)
   obj.mine()
   obj.save('mffiPatternsFromURL.txt')

The URL must return text in the same format as a local file. This dataset uses
the default tab separator.

Running from a terminal
-----------------------

From an installed source checkout, run:

.. code-block:: console

   python3 PAMI/fuzzyFrequentPattern/basic/MFFIMiner.py <inputFile> <outputFile> <minSup> [sep]

For the space-separated sample:

.. code-block:: console

   python3 PAMI/fuzzyFrequentPattern/basic/MFFIMiner.py fuzzyTransactions.txt mffiPatterns.txt 1 ' '

Omit the separator argument for a tab-separated database. An integer argument
such as ``1`` sets an absolute threshold; ``0.25`` sets a proportion.

Runtime and memory
------------------

``getRuntime()`` returns elapsed mining time in seconds.
``getMemoryRSS()`` and ``getMemoryUSS()`` return process memory in bytes,
measured after mining. These values are neither peak memory nor the additional
memory used only by the algorithm. ``printResults()`` prints the pattern count
and these measurements.

Limitations
-----------

* Input must already contain fuzzy memberships. MFFIMiner does not transform
  quantitative data into fuzzy data.
* Each row must use ``items:memberships``. A third field containing a total
  utility or total membership is not accepted.
* Item and membership lists must have equal lengths. Empty labels, duplicate
  fuzzy labels, and memberships outside ``[0, 1]`` are rejected.
* Region labels must use a consistent ``item.Region`` convention to enforce
  the one-region-per-base-item rule.
* Retaining all frequent regions can produce many patterns at low minimum
  support thresholds, increasing runtime and memory use.

API
---

.. automodule:: PAMI.fuzzyFrequentPattern.basic.MFFIMiner
   :members:
   :undoc-members:
   :show-inheritance:
