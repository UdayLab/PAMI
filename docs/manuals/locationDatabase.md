[__<--__ Previous ](uncertainDatabases.html)|[Home](index.html)|[_Next_-->](neighborhoodDatabase.html)

## Location/Spatial database

### Description
A spatial database is a collection of spatial objects (or items), such as pixels, points, lines, and polygons.
A sample spatial database generated from the set of items, I={a,b,c,d,e,f}, is shown in below table:

Item |  Spatial information
  --- | -----
a   | Point(0 0)
b   | Point(0 1)
c   | Point(1 0)
d   | Point(0 2)
e   | Point(4 0)
f   | Point(5 1)

### Rules to create a location database

1. Every row must contain only two columns, namely _item_ and its _spatial information_.
2. An item and its spatial information have to be seperated using a seperator, such as tab space.


### Format of a location database

         item<sep>spatialInformation

### An example
    a   Point(0 0)
    b   Point(0 1)
    c   Point(1 0)
    d   Point(0 2)
    e   Point(4 0)
    f   Point(5 1)


