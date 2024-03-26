Fuzzy Database
================

    A fuzzy database represents the data generated from a non-binary transactional or temporal database using fuzzy logic.

    Types

        - Fuzzy transactional databases

        - Fuzzy temporal databases

    Fuzzy transactional databases

        A fuzzy transactional database represents a set of transactions, where each transaction consists of a transactional identifier (tid), items, and their fuzzy occurrences values. Please note that the fuzzy occurrence values of an item lie between 0 and 1. If the fuzzy value of an item is close zero, it implies less chance of occurrence of an item in a database. If the fuzzy value of an item is close one, it implies high chance of occurrence of an item in a database. A sample fuzzy transactional database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat}, is shown in below table:

        +--------+--------------------------------------------------------------------------------------------------------------+
        | TID    | Transactions (items and their fuzzy values)                                                                  |
        +========+==============================================================================================================+
        | 1      | (Bread.High, 0.6), (Bread.Low, 0.4), (Jam.High, 0.2), (Jam.Low, 0.8), (Butter.High, 0.8), (Butter.Low, 0.2)  |
        +--------+--------------------------------------------------------------------------------------------------------------+
        | 2      | (Bat.High, 0.5), (Bat.Low, 0.5), (Ball.High, 0.6), (Ball.Low, 0.4)                                           |
        +--------+--------------------------------------------------------------------------------------------------------------+
        | 3 	 | (Pen.High, 0.2), (Pen.Low, 0.8), (Book.High, 0.3), (Book.Low, 0.7)                                           |
        +--------+--------------------------------------------------------------------------------------------------------------+

    Format of a fuzzy transactional database

        The fuzzy transactional database must exist in the following format:

            >>> fuzzyitemA<sep>fuzzyitemB<sep>...<sep>fuzzyitemN:total fuzzyValue:fuzzyValueA<sep>fuzzyValueB<sep>...<sep>fuzzyValueN

            The ‘total fuzzy value’ represents the sum of fuzzy values of all items in a transaction.

    Rules to create a fuzzy database

        - The default separator, i.e., , used in PAMI is tab space (or \t). However, the users can override the default separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.

        - Items, total utility, and individual utilities of the items within a transaction have to be seperated by the symbol ‘:’

    An example

        Bread.High    Bread.Low   Jam.High    Jam.Low     Butter.High    Butter.Low:3:0.6    0.4    0.2    0.8    0.8   0.2

        Bat.High   Bat.Low   Ball.High   Ball.Low:2:0.5    0.5    0.6    0.4

        Pen   Book:2:0.2    0.8   0.3    0.7

    Fuzzy temporal databases

        A fuzzy temporal database consists of timestamp, tid, items, and their corresponding fuzzy values. A sample fuzzy temporal database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat}, is shown in below table:

        +------------+---------+---------------------------------------------------------------------------------------------------------------+
        | Timestamp  | tid     | Transactions (items and their fuzzy values)                                                                   |
        +============+=========+===============================================================================================================+
        | 1          | 1       | (Bread.High, 0.6), (Bread.Low, 0.4), (Jam.High, 0.2), (Jam.Low, 0.8), (Butter.High, 0.8), (Butter.Low, 0.2)   |
        +------------+---------+---------------------------------------------------------------------------------------------------------------+
        | 2          | 2       | (Bat.High, 0.5), (Bat.Low, 0.5), (Ball.High, 0.6), (Ball.Low, 0.4)                                            |
        +------------+---------+---------------------------------------------------------------------------------------------------------------+
        | 5          | 3       | (Pen.High, 0.2), (Pen.Low, 0.8), (Book.High, 0.3), (Book.Low, 0.7)                                            |
        +------------+---------+---------------------------------------------------------------------------------------------------------------+

    Format of fuzzy temporal database

        The fuzzy temporal database must exist in the following format:

            >>> timestamp:fuzzyitemA<sep>fuzzyitemB<sep>...<sep>fuzzyitemN:total fuzzy value:fuzzyValueA<sep>fuzzyValueB<sep>...<sep>fuzzyValueN

            The ‘total fuzzy value’ represents the total fuzzy value of all items in a transaction.

    Rules to create a fuzzy temporal database

        - The default separator, i.e., , used in PAMI is tab space (or \t). However, the users can override the default separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.

        - Timestamp, items, total utility, and individual utilities of the items within a transaction have to be seperated by the symbol ‘:’

    An example

        1:Bread   Jam     Butter:3:0.6    0.4    0.2    0.8    0.8   0.2

        2:Bat Ball:110:100   10

        5:Pen Book:7:2 5





.. toctree::
   :maxdepth: 1

   FuzzyFrequentPatternMining1

.. toctree::
   :maxdepth: 1

   FuzzyCorrelatedPatternMining1

.. toctree::
   :maxdepth: 1

   FuzzyGeoReferencedFrequentPatternMining1

.. toctree::
   :maxdepth: 1

   FuzzyPeriodicFrequentPatternMining1

.. toctree::
   :maxdepth: 1

   FuzzyGeoReferencedPeriodicFrequentPatternMining1


