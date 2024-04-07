Utility Database
==================

    A transactional/temporal database represents a binary database. It is because the items in these databases have values either 1 or 0. In contrast, a utility database is a non-binary database. In fact, a utility database is a quantitative database containing items and their utility values.

    Utility databases are naturally produced by the real-world applications. Henceforth, most forms of the databases, such as transactional and temporal databases, are generated from utility databases.

    In the utility database, the items have external utility values and internal utility values. External utility values, like prices of items in a supermarket, do not vary in the entire data. Internal utility values, like the number of items purchased by each customer, vary for every transaction in the database. The utility of an item in a transaction represents the product of its internal and external utility values.

    Types

        - Utility transactional databases

        - Utility temporal databases

    Utility transactional databases

    Description

        A utility transactional database consists of a transactional identifier (tid), items, and their corresponding utility values in a transaction. A sample utility transactional database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat}, is shown in below table:

        +------+------------------------------------------+
        | TID  | Transactions (items and their prices)    |
        +======+==========================================+
        | 1    | (Bread,1$), (Jam,2$), (Butter, 1.5$)     |
        +------+------------------------------------------+
        | 2    | (Bat, 100$), (Ball, 10$)                 |
        +------+------------------------------------------+
        | 3    | (Pen, 2$), (Book, 5$)                    |
        +------+------------------------------------------+

    Format of a utility transactional database

        The utility transactional database must exist in the following format:

            >>> itemA<seo>itemB<sep>...<sep>itemN : total utility : utilityA<sep>utilityB<sep>...<sep>utilityN

            The ‘total utility’ represents the total utility value of items in a transaction.

    Rules to create a utility transactional database

        - The default separator, i.e., , used in PAMI is tab space (or \t). However, the users can override the default separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.

        - Items, total utility, and individual utilities of the items within a transaction have to be seperated by the symbol ‘:’

    An example:

        +-------------------+------------------+
        |Bread Jam Butter:  | 4.5:1    2   1.5 |
        +-------------------+------------------+
        |Bat Ball:          | 110:100   10     |
        +-------------------+------------------+
        |Pen Book:          | 7:2   5          |
        +--------------------+-----------------+

    Utility temporal databases

    Description

        A utility temporal database consists of timestamp, tid, items, and their corresponding utility values. A sample utility temporal database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat}, is shown in below table:

        +-----------+---------+--------------------------------------------+
        | Timestamp | tid     |	Transactions (items and their prices)      |
        +===========+=========+============================================+
        | 1         | 1 	  | (Bread,1$), (Jam,2$), (Butter, 1.5$)       |
        +-----------+---------+--------------------------------------------+
        | 2 	    | 2       | (Bat, 100$), (Ball, 10$)                   |
        +-----------+---------+--------------------------------------------+
        | 5         | 3       | (Pen, 2$), (Book, 5)                       |
        +-----------+---------+--------------------------------------------+

    Format of a utility temporal database

        The utility temporal database must exist in the following

    Format:

       >>>  timestamp:itemA<seo>itemB<sep>...<sep>itemN:total utility:utilityA<sep>utilityB<sep>...<sep>utilityN

       The ‘total utility’ represents the total utility value of items in a transaction.

    Rules to create a utility temporal database

        - The default separator, i.e., , used in PAMI is tab space (or \t). However, the users can override the default separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.

        - Timestamp, items, total utility, and individual utilities of the items within a transaction have to be seperated by the symbol ‘:’

    Example:

        +---+-------------------+------------------+
        | 1 |Bread Jam Butter:  | 4.5:1    2   1.5 |
        +---+------------------+-------------------+
        | 2 |Bat Ball:          | 110:100   10     |
        +---+-------------------+------------------+
        | 3 |Pen Book:          | 7:2   5          |
        +---+-------------------+------------------+


.. toctree::
   :maxdepth: 1

   HighUtilityPatternMining1

.. toctree::
   :maxdepth: 1

   HighUtilityFrequentPatternMining1

.. toctree::
   :maxdepth: 1

   HighUtilityGeo-referencedFrequentPatternMining1


.. toctree::
   :maxdepth: 1

   HighUtilitySpatialPatternMining1


.. toctree::
   :maxdepth: 1

   RelativeHighUtilityPatternMining1

.. toctree::
   :maxdepth: 1

   WeightedFrequentPatternMining1

.. toctree::
   :maxdepth: 1

   WeightedFrequentRegularPatternMining1


.. toctree::
   :maxdepth: 1

   WeightedFrequentNeighbourhoodPatternMining1
