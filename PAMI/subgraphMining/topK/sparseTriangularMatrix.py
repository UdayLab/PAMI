class SparseTriangularMatrix:
    """
    Represents a sparse triangular matrix structure for storing counts or support values.

    Attributes:
        matrix (dict): A dictionary representing the sparse matrix structure.

    """
    def __init__(self):
        """
        Initializes the SparseTriangularMatrix object.
        """
        self.matrix = {}  

    def __str__(self):
        """
        Returns a string representation of the matrix.
        """
        temp = []
        for key in sorted(self.matrix.keys()):
            subkeys = self.matrix[key]
            subkeyStr = " ".join(f"{subkey}:{count}" for subkey, count in subkeys.items())
            temp.append(f"{key}: {subkeyStr}\n")
        return "".join(temp)

    def incrementCount(self, i, j):
        """
        Increments the count or support value at the position (i, j) in the matrix.
        """
        if i < j:
            key, subkey = i, j
        else:
            key, subkey = j, i

        if key not in self.matrix:
            self.matrix[key] = {subkey: 1}
        else:
            if subkey not in self.matrix[key]:
                self.matrix[key][subkey] = 1
            else:
                self.matrix[key][subkey] += 1

    def getSupportForItems(self, i, j):
        """

        Retrieves the support value for items at positions (i, j) in the matrix.

        """
        smaller, larger = min(i, j), max(i, j)
        return self.matrix.get(smaller, {}).get(larger, 0)

    def setSupport(self, i, j, support):
        """
        Sets the support value for items at positions (i, j) in the matrix.

        """
        smaller, larger = min(i, j), max(i, j)
        
        if smaller not in self.matrix:
            self.matrix[smaller] = {larger: support}
        else:
            self.matrix[smaller][larger] = support

    def removeInfrequentEntriesFromMatrix(self, minsup):
        """
        Removes entries from the matrix with support values below a specified minimum support threshold.
        """
        for key in list(self.matrix.keys()):
            for subkey, value in list(self.matrix[key].items()):
                if value < minsup:
                    del self.matrix[key][subkey]
            if not self.matrix[key]:
                del self.matrix[key]
