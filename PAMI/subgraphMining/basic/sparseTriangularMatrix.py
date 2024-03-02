class SparseTriangularMatrix:
    def __init__(self):
        self.matrix = {}  

    def __str__(self):
        temp = []
        for key in sorted(self.matrix.keys()):
            subkeys = self.matrix[key]
            subkeyStr = " ".join(f"{subkey}:{count}" for subkey, count in subkeys.items())
            temp.append(f"{key}: {subkeyStr}\n")
        return "".join(temp)


    # def incrementCount(self, i, j):
    #     smaller, larger = min(i, j), max(i, j)

    #     if smaller not in self.matrix:
    #         self.matrix[smaller] = {larger: 1}
    #     else:
    #         self.matrix[smaller][larger] = self.matrix[smaller].get(larger, 0) + 1

    def incrementCount(self, i, j):
        # print(f"Incrementing count for pair ({i}, {j})")
        if i < j:
            key, subkey = i, j
        else:
            key, subkey = j, i

        if key not in self.matrix:
            self.matrix[key] = {subkey: 1}
            # print(f"New pair added: {key}-{subkey}, count: 1")
        else:
            if subkey not in self.matrix[key]:
                self.matrix[key][subkey] = 1
                # print(f"New subkey added for existing key: {key}-{subkey}, count: 1")
            else:
                self.matrix[key][subkey] += 1
                # print(f"Count updated for {key}-{subkey}, new count: {self.matrix[key][subkey]}")

    def getSupportForItems(self, i, j):
        smaller, larger = min(i, j), max(i, j)
        return self.matrix.get(smaller, {}).get(larger, 0)

    def setSupport(self, i, j, support):
        smaller, larger = min(i, j), max(i, j)
        
        if smaller not in self.matrix:
            self.matrix[smaller] = {larger: support}
        else:
            self.matrix[smaller][larger] = support

    def removeInfrequentEntriesFromMatrix(self, minsup):
        for key in list(self.matrix.keys()):
            for subkey, value in list(self.matrix[key].items()):
                if value < minsup:
                    del self.matrix[key][subkey]
            if not self.matrix[key]:
                del self.matrix[key]
