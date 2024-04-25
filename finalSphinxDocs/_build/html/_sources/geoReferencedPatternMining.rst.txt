Geo-referenced Pattern Mining
==============================

    A geo-referenced database represents the data gathered by a set of fixed sensors observing a particular phenomenon over a time period. It is a combination of spatial database and transactional/temporal/utility database .

    Types of Geo-referenced databases

        - Geo-referenced transactional databases

        - Geo-referenced temporal databases

        - Geo-referenced utility database

    Basic topics

        - Location/spatial database

        - Neighborhood database

    1. Geo-referenced transactional database

        A transactional database is said to be a geo-referenced transactional database if it contains spatial items. The format of this database is similar to that of transactional database . An example of a geo-referenced transactional database is as follows:

        +------+--------------------------------------+
        | TID  | Items                                |
        +======+======================================+
        | 1    | Point(0 0) Point(0 1) Point(1 0)     |
        +------+--------------------------------------+
        | 2    | Point(0 0) Point(0 2) Point(5 0)     |
        +------+--------------------------------------+
        | 3    | Point(5 0)                           |
        +------+--------------------------------------+
        | 4    | Point(4 0) Point(5 0)                |
        +------+--------------------------------------+

        Note: The rules to create a geo-referenced transactional database are same as the rules to create a transactional database. In other words, the format of creating a transaction in a geo-referential database is:

                >>> spatialItem1<sep>spatialItem2<sep>...<sep>spatialItemN

        An example:

            Point(0 0)    Point(0 1)  Point(1 0)

            Point(0 0)    Point(0 2)  Point(5 0)

            Point(5 0)

            Point(4 0)    Point(5 0)


    2. Geo-referential temporal database

        A temporal database is said to be a geo-referential temporal database if it contains spatial items. The format of this database is similar to that of temporal database . An example of a geo-referential temporal database is as follows:

        +-----+------------+----------------------------------------+
        | TID | Timestamp  | Items                                  |
        +=====+============+========================================+
        | 1   |	1          | Point(0 0) Point(0 1) Point(1 0)       |
        +-----+------------+----------------------------------------+
        | 2   |	2          | Point(0 0) Point(0 2) Point(5 0)       |
        +-----+------------+----------------------------------------+
        | 3   |	4          | Point(5 0)                             |
        +-----+------------+----------------------------------------+
        | 4   | 5          | Point(4 0) Point(5 0)                  |
        +-----+------------+----------------------------------------+

        Note: The rules to create geo-referential temporal database are same as the rules to create a temporal database. In other words, the format to create geo-referential temporal database is as follows:

            >>> timestamp<sep>spatialItem1<sep>spatialItem2<sep>...<sep>spatialItemN

        An example:

            1   Point(0 0)    Point(0 1)  Point(1 0)

            2   Point(0 0)    Point(0 2)  Point(5 0)

            4   Point(5 0)

            5   Point(4 0)    Point(5 0)

    3. Geo-referential utility database

        A utility database is said to be a geo-referential utility database if it contains spatial items. The format of this database is similar to that of utility database . An example of a geo-referential utility database is as follows:

        +-------+----------------------------------------------------+
        | TID   | Transactions (items and their prices)              |
        +=======+====================================================+
        | 1     | (Point(0 0),100) (Point(0 1),42) (Point(1 0), 20)  |
        +-------+----------------------------------------------------+
        | 2     | (Point(0 0), 100) (Point(0 2), 10) (Point(5 0), 30)|
        +-------+----------------------------------------------------+
        | 3     | (Point(5 0), 30)                                   |
        +-------+----------------------------------------------------+
        | 4     | (Point(4 0),30), (Point(5 0),40)                   |
        +-------+----------------------------------------------------+

    Note: The rules to create geo-referential utility database are same as the rules to create a utility database. In other words, the format to create geo-referential utility database is as follows:

    >>> timestamp<sep>spatialItem1<sep>spatialItem2<sep>...<sep>spatialItemN : total utility : utilityA<sep>utilityB<sep>...<sep>utilityN

    An example:

            1   Point(0 0)    Point(0 1)  Point(1 0):162:100    42  20

            2   Point(0 0)    Point(0 2)  Point(5 0):140:100    10  30

            4   Point(5 0):30:30

            5   Point(4 0)    Point(5 0):70:30  40



.. toctree::
   :maxdepth: 1

   GeoReferencedFrequentPatternMining1


.. toctree::
   :maxdepth: 1

   GeoReferencedPeriodicFrequentPatternMining1


.. toctree::
   :maxdepth: 1

   GeoReferencedPartialPeriodicPatternMining1
