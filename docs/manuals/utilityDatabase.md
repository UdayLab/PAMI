[Previous](temporalDatabase.html)|[Home](index.html)|[Next](uncertainDatabases.html)


## Utility databases

### Description

A utility database represents a non-binary transactional or temporal database.

### Types
1. Utility transactional databases
2. Utility temporal databases

### Utility transactional databases
#### Introduction
A utility transactional database consists of a transactional identifier (tid), items, and their corresponding utility values in a transaction.
A sample utility transactional database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat},
is shown in below table:

  TID |  Transactions (items and their prices)
     --- | -----
     1   | (Bread,1$), (Jam,2$), (Butter, 1.5$)
     2   | (Bat, 100$), (Ball, 10$)
     3   | (Pen, 2$), (Book, 5$) 

#### Format of a utility transactional database
The utility transactional database must exist in the following format:

     itemA<seo>itemB<sep>...<sep>itemN:total utility:utilityA<sep>utilityB<sep>...<sep>utilityN

_The 'total utility' represents the total utility value of items in a transaction._


#### Rules to create a utility transactional database

1. The default separator, i.e., <sep>, used in PAMI is tab space (or \t). However, the users can override the default 
   separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space 
   and comma, usage of tab space facilitates us to effectively distinguish the spatial objects. 
2. Items, total utility, and individual utilities of the items within a transaction have to be seperated by the symbol ':'

#### An example

      Bread   Jam     Butter:4.5:1    2   1.5

      Bat   Ball:110:100   10

      Pen   Book:7:2   5

### Utility temporal databases
#### Introduction
An utility temporal database consists of timestamp, tid, items, and their corresponding utility values. 
A sample utility temporal database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat},
is shown in below table:

  Timestamp | tid| Transactions (items and their prices)
     --- | -----|----
    1| 1   | (Bread,1$), (Jam,2$), (Butter, 1.5$)
    2| 2   | (Bat, 100$), (Ball, 10$)
    5| 3   | (Pen, 2$), (Book, 5$) 

#### Format of a utility temporal database
The utility temporal database must exist in the following format:

     timestamp:itemA<seo>itemB<sep>...<sep>itemN:total utility:utilityA<sep>utilityB<sep>...<sep>utilityN

_The 'total utility' represents the total utility value of items in a transaction._

#### Rules to create a utility temporal database
1.  The default separator, i.e., <sep>, used in PAMI is tab space (or \t). However, the users can override the default 
   separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space 
   and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.
    
1. Timestamp, items, total utility, and individual utilities of the items within a transaction have to be seperated by the symbol ':'

#### An example

      1:Bread   Jam     Butter:4.5:1    2   1.5

      2:Bat Ball:110:100   10

      5:Pen Book:7:2 5
