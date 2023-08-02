[__<--__ Return to home page](index.html)

## Uncertain database

### Description
An uncertain database is a non-binary database, where an occurrence of an item in a transaction is associated with a 
probabilistic value that lies between zero and one. The value zero represents the complete non-occurrence of an item, while the 
value represents the perfect occurrence of an item in a transaction.

Currently, the algorithms in PAMI support the discovery of knowledge hidden in two types of uncertain databases, namely uncertain transactional database and uncertain temporal database.
We now describe each of these databases.

### Types
1. Uncertain transactional database
2. Uncertain temporal database

## Uncertain transactional database
### Introduction
An uncertain transactional database consists of a transactional identifier (tid), items, and their occurrence probability value.
A sample uncertain transactional database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat},
is shown in below table:

TID |  Transactions (items and their prices)
     --- | -----
1   | (Bread,0.9), (Jam,0.7), (Butter, 0.1)
2   | (Bat, 1), (Ball, 0.5)
3   | (Pen, 0.2), (Book, 0.5) 

__Note:__ The above uncertain database represents an uncertain transactional database. If every transaction in an uncertain database
is associated with a timestamp, then we call that database an uncertain temporal database.

### Format to create uncertain transactional databases in PAMI
An utility transactional database must exist in the following format:

     itemA<sep>itemB<sep>...<sep>itemN:total probability:probabilityA<sep>probabilityB<sep>...<sep>probabilityN

_The 'total probability' represents the sum of probabilities of all items in a transaction._

### Rules to create a uncertain transactional databases
1. The default separator, i.e., <sep>, used in PAMI is tab space (or \t). However, the users can override the default
    separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space
    and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.

2. Items, total probability, and individual probabilities of the items within a transaction have to be seperated by the symbol ':'
3. The probability values of an item must be within the range [0,1]. 

### An example of an uncertain transactional database

      Bread   Jam     Butter:1.7:0.9    0.7 0.1

      Bat   Ball:1.5:1  0.5

      Pen   Book:0.7:0.2    0.5
## Uncertain temporal database
### Introduction
An uncertain temporal database consists of a transactional identifier (tid), a timestamp, items, and their occurrence probability value.
A sample uncertain temporal database generated from the set of items, I={Bread, Jam, Butter, Pen, Books, Bat},
is shown in below table:

TID | TS  | Transactions (items and their prices)
     --- |-----| -----
1   | 1   |(Bread,0.9), (Jam,0.7), (Butter, 0.1)
2   | 4   |(Bat, 1), (Ball, 0.5)
3   | 5   |(Pen, 0.2), (Book, 0.5) 

### Format to create an uncertain temporal databases in PAMI
An utility temporal database must exist in the following format:

     timestamp<sep>itemA<sep>itemB<sep>...<sep>itemN:total probability:probabilityA<sep>probabilityB<sep>...<sep>probabilityN

_The 'total probability' represents the sum of probabilities of all items in a transaction._

### Rules to create an uncertain temporal databases
1. First element in every transaction must be a timestamp.
2. The default separator, i.e., <sep>, used in PAMI is tab space (or \t). However, the users can override the default
   separator with their choice. Since spatial objects, such as Point, Line, and Polygon, are represented using space
   and comma, usage of tab space facilitates us to effectively distinguish the spatial objects.

3. Items, total probability, and individual probabilities of the items within a transaction have to be seperated by the symbol ':'
4. The probability values of an item must be within the range [0,1].

### An example of an uncertain temporal database

      1 Bread   Jam     Butter:1.7:0.9    0.7 0.1

      2 Bat   Ball:1.5:1  0.5

      3 Pen   Book:0.7:0.2    0.5