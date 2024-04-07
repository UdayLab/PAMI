Sequential Database
=====================

    A sequence represents a collection of itemsets (or transactions) in a particular order. A sequence database is a collection of sequences and their sequence identifiers. An example of a geo-referenced transactional database is as follows:

    +--------+----------------------------------+
    | SID 	 |  Items                           |
    +========+==================================+
    | 1 	 |  {a b c d} {a d e} {a e f}       |
    +--------+----------------------------------+
    | 2      | 	{a b c} {b d e} {c d e}         |
    +--------+----------------------------------+
    | 3      | 	{a e f} {c d}                   |
    +--------+----------------------------------+
    | 4      | 	{a e f} {a c d} {c e}           |
    +--------+----------------------------------+

    Rules to create a sequence database:

        - Items in an itemset have to be seperated by a tab space.

        - Itemsets in a sequence are seperated using '-1' as a seperator.

        - Each sequence is represented as a line

        - The sequence identifier, sid, is not needed to create a sequence database.

Format of a sequence:

            >>> item1<sep>item2<sep>...<sep>itemA : item1<sep>item2<sep>...<sep>itemB : item1<sep>item2<sep>...<sep>itemC

Example:

    >>> a b c d : a d e : a e f
        a b c : b d e : c d e
        a e f : c
        a e f : a c d : c e




.. toctree::
   :maxdepth: 1

   SequentialFrequentPatternMining1


.. toctree::
   :maxdepth: 1

   GeoReferencedFrequentSequencePatternMining1


