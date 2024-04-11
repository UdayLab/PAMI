[__<--__ Previous ](organization.html)|[Home](index.html)|[_Next_-->](temporalDatabase.html)

## Transactional database

### Description
A transactional database is a collection of transactions.  Every transaction constitutes of a transaction-identifier (TID)
and a set of items. A sample transactional database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:

  TID |  Transactions 
     --- | -----
     1   | a, b, c
     2   | d, e
     3   | a, e, f
   
### Rules to create a transactional database

1. Since TID of a transaction directly represents its row number in a database, we can ignore this information 
to save storage space and processing time.
2. The items in a transactional database can be integers or strings.
3. All items in a transaction must be seperated with a separator. 
4. '_Tab space_' is the default seperator.   However, transactional databases can be constructed using other seperators, such as comma and space.

### Format of a transactional database
 
    item1<sep>item2<sep>...<sep>itemN

### An example
    a   b   c
    d   e
    a   e   f
