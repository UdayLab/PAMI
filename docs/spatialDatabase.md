# **[Home](index.html) | [Exercises](exercises.html) | [Real-world Examples](examples.html)**  

# Spatial database

A spatial database is a collection of spatial objects (or items), such as pixels, points, lines, and polygons. 
   The format of the spatial database is as follows:
   
         item : spatialInformation (pixel, point, line, polygon)

   A sample spatial database for the set of items, I={a,b,c,d,e,f}, is shown in below table.
  
 Item |  Spatial information 
  --- | -----
  a   | Point(0 0)
  b   | Point(0 1)
  c   | Point(1 0)
  d   | Point(0 2)
  e   | Point(4 0)
  f   | Point(5 1)


## Spatiotransactional database
A transactional database is said to be a spatiotemporal database if its items represents spatial items. An example of a 
spatiotransactional database is as follows:

TID | Items
--- | -----
 1  | Point(0 0)    Point(0 1)  Point(1 0)
 2  | Point(0 0)    Point(0 2)  Point(5 0)
 3  | Point(5 0)
 4  | Point(4 0)    Point(5 0)
 
### Rules to create spatiotransactional database 
Rules to create the spatiotransactional database are similar to the rules to create [transactional database](transactionalDatabase.html). 
The rules are as follows:
1. Since TID of a transaction directly represents its row number in a database, we can ignore this information 
to save storage space and processing time. 
1. All spatial items in every transaction must be with a separator.   
 
 
 ### Spatiotransactional database format
The format of a spatiotransactional database is as follows:
    
    spatialItem1<sep>spatialItem2<sep>...<sep>spatialItemN
    
An example:

    Point(0 0)    Point(0 1)  Point(1 0)
    Point(0 0)    Point(0 2)  Point(5 0)
    Point(5 0)
    Point(4 0)    Point(5 0)


#Spatiotemporal database
A temporal database is said to be a spatiotemporal database if its items represents spatial items. An example of a 
spatiotemporal database is as follows:

TID | Timestamp | Items
--- | --------- | -----
 1  | 1 | Point(0 0)    Point(0 1)  Point(1 0)
 2  | 2 | Point(0 0)    Point(0 2)  Point(5 0)
 3  | 4 | Point(5 0)
 4  | 5 | Point(4 0)    Point(5 0)

### Rules to create spatiotemporal database 
Rules to create the spatiotemporal database are similar to the rules to create [temporal database](temporalDatabase.html). 
The rules are as follows:
1. Since TID of a transaction implicitly represents the row number, this information can be ignored to save space.
1. The first column of every transaction must represent a timestamp. 
1. The timestamp of the first transaction must always be 1. The timestamps of remaining transactions follow thereafter. 
   In other words, the timestamps in a temporal database must be relative to each other, rather than absolute timestamps.
1. Irregular time gaps can exist between the transactions.
1. Multiple transactions can have a same timestamp. In other words, multiple transactions can occur at a particular timestamp.


 Please refer to [temporal database](temporalDatabase.html) to get more details on a temporal database. 
 
### Spatiotemporal database format
The format of a spatiotemporal database is as follows:
    
    timestamp<sep>spatialItem1<sep>spatialItem2<sep>...<sep>spatialItemN
    
An example:

    1   Point(0 0)    Point(0 1)  Point(1 0)
    2   Point(0 0)    Point(0 2)  Point(5 0)
    4   Point(5 0)
    5   Point(4 0)    Point(5 0)