# What is a transactional database?

A transactional database is a collection of transactions.  Every transaction constitutes of a transaction-identifier (TID)
and a set of items. A sample transactional database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:

  TID |  Transactions 
     --- | -----
     1   | a, b, c
     2   | d, e
     3   | a, e, f
   
# What are the rules to be followed in creating a transactional database?

1. Since TID of a transaction directly represents its row number in a database, we can ignore this information 
to save storage space and processing time.
2. The items in a transactional database can be integers or strings.
3. All items in a transaction must be seperated with a separator. 
4. '_Tab space_' is the default seperator.   However, transactional databases can be constructed using other seperators, such as comma and space.

Overall, the format of a transaction in a transactional database is as follows:

      item1<sep>item2<sep>...<sep>itemN

# Given an example of a transactional database?

  An example of a transactional database is as follows:

        a   b   c
        d   e
        a   e   f
