Temporal Database
=========================

        A temporal database is a collection of transactions ordered by their timestamps. A sample temporal database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:

            +-----+--------------+--------------------+
            |TID  | Timestamp    | Transactions       |
            +=====+==============+====================+
            | 1   | 1            | a, b, c            |
            +-----+--------------+--------------------+
            | 2   | 2            | d, e               |
            +-----+--------------+--------------------+
            | 3   | 4            | a, e, f            |
            +-----+--------------+--------------------+
            |4    | 7            | d, f, g            |
            +-----+--------------+--------------------+

        Types of temporal databases:

            - Regular temporal database: Uniform time gap exists between any two transactions.

            - Irregular temporal database: Non-uniform time gap exists between any two transactions.

                - Type-1 irregular temporal database: Every transaction will have a distinct timestamp.

                - Type-2 irregular temporal database: Multiple transactions can have a common timestamp.

        Rules to create a temporal database:

            - Since TID of a transaction implicitly represents the row number, this information can be ignored to save space.

            - The first column in the database must represent a timestamp.

            - The timestamp of the first transaction must always start from 1. The timestamps of remaining transactions follow thereafter. In other words, the timestamps in a temporal database must be relative to each other, rather than being absolute timestamps.

            - Irregular time gaps can exist between the transactions.

            - Multiple transactions can have a same timestamp. In other words, multiple transactions can occur at a particular timestamp. (Please note that some pattern mining algorithms, especially variants of ECLAT, may not work properly if multiple transactions share a common timestamp.)

            - All items in a transaction must be seperated with a separator.

            - The items in a temporal database can be integers or strings.

            - ‘ Tab space ’ is the default seperator. However, temporal databases can be constructed using other seperators, such as comma and space.

        Format of a temporal database:

                 >>>  timestamp<sep>item1<sep>item2<sep>...<sep>itemN

        Examples:

        - Regular temporal database: Uniform time gap exists between the transactions.

            1   a   b   c

            2   d   e

            4   a   e   f

            7   d   f   g


        - Irregular temporal database (Type-1): Non-uniform time gap exists between the transactions. More important, every transaction contains a unique timestamp.

            1   a   b   c

            2   d   e

            4   a   e   f

            7   d   f   g


        - Irregular temporal database (Type-2): Non-uniform time gap exists between the transactions. More important, multiple transactions can have same timestamps.

            1   a   b   c

            1   d   e

            4   a   e   f

            8   d   f   g




.. toctree::
   :maxdepth: 1

   PeriodicFrequentPatternMining1

.. toctree::
   :maxdepth: 1

   LocalPeriodicPatternMining1


.. toctree::
   :maxdepth: 1

   PartialPeriodicFrequentPatternMining1


.. toctree::
   :maxdepth: 1

   PartialPeriodicPatternMining1


.. toctree::
   :maxdepth: 1

   PeriodicCorrelatedPatternMining1


.. toctree::
   :maxdepth: 1

   StablePeriodicPatternMining1


.. toctree::
   :maxdepth: 1

   RecurringPatternMining1

