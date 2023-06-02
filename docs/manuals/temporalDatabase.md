[__<--__ Return to home page](index.html)

## Temporal database

### Description
 A temporal database is a collection of transactions ordered by their timestamp. A sample temporal database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:
   
   TID |  Timestamp | Transactions 
     --- | ----- | ---
     1  | 1  | a, b, c
     2  | 2 | d, e
     3  | 5 | a, e, f
     4  | 5 | d, f, g  

### Rules to create a temporal database

1. Since TID of a transaction implicitly represents the row number, this information can be ignored to save space.
2. The first column in the database must represent a timestamp.
3. The timestamp of the first transaction must always start from 1. The timestamps of remaining transactions follow thereafter. __In other words, the timestamps in a temporal database must be relative to each other, rather than being absolute timestamps.__
4. Irregular time gaps can exist between the transactions.
5. Multiple transactions can have a same timestamp. In other words, multiple transactions can occur at a particular timestamp. (Please note that some pattern mining algorithms, especially variants of ECLAT, may not work properly if multiple transactions share a common timestamp.)
6. All items in a transaction must be seperated with a separator.
7. The items in a temporal database can be integers or strings.
8. '_Tab space_' is the default seperator.   However, temporal databases can be constructed using other seperators, such as comma and space.

### Format of a temporal database

    timestamp<sep>item1<sep>item2<sep>...<sep>itemN

# An example
    1   a   b   c
    2   d   e
    5   a   e   f
    5   d   f   g


