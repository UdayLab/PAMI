 A transactional database is a set of transactions.
    Each transaction contains a transaction-identifier (TID) and a set of items.

Example:


    A sample transactional database containing the items from a to f is shown in below.

    +-----+----------------+
    | TID | Transactions   |
    +=====+================+
    | 1   |  a, b, c       |
    +-----+----------------+
    | 2   |  d, e          |
    +-----+----------------+
    | 3   |  a, e, f       |
    +-----+----------------+

Rules to create a transactional database:

    -Since the TID of a transaction directly represents its row number in a database, we the algorithms in PAMI ignore the TID information to save storage space and processing time.
    -The items in a transactional database can be integers or strings.
    -All items in a transaction must be seperated with a separator.
    -‘ Tab space ’ is the default seperator used by the mining algorithms in PAMI. However, transactional databases can also be constructed using other separators, such as comma and space.

Format:

        >>>  item1<sep>item2<sep>...<sep>itemN

Example:

     >>>    a   b   c
            a   d   e   f
            b   d


Frequent Pattern mining
=========================

Frequent pattern mining is the process of identifying patterns or associations within a dataset that occur frequently.
This is typically done by analyzing large datasets to find items or sets of items that appear together frequently.

Applications: DNA sequences, protein structures, leading to insights in genetics and drug design.


.. toctree::
   :hidden:
   :caption: Frequent Pattern Mining:

   frequentPatternMining


Relative Frequent Pattern
==========================

Relative Frequent Pattern definition here

.. toctree::
   :hidden:
   :caption: Relative Frequent Pattern Mining:

   relativeFrequentPattern

Frequent pattern With Multiple Minimum Support
===============================================

Frequent Pattern with Multiple Support definition here

.. toctree::
   :hidden:
   :caption: Frequent Pattern With Multiple Minimum Support:

   frequentPatternWithMultipleMinimumSupport

Correlated Pattern Mining
==========================

Correlated Pattern Mining definition here

.. toctree::
   :hidden:
   :caption: Correlated Pattern Mining:

   correlatedPatternMining

Fault-Tolerant Frequent Pattern Mining
========================================

Fault Tolerant Pattern Mining definition here

.. toctree::
   :hidden:
   :caption: Fault-Tolerant Frequent Pattern Mining:

   faultTolerantPatternMining

Coverage Pattern Mining
==========================

Coverage Pattern Mining definition here

.. toctree::
   :hidden:
   :caption: Coverage Pattern Mining:

   coveragePatternMining
