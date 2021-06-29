class UPNode:
    # item name
    itemId = -1
    count = 1
    # utility of this node
    nodeUtility = 0
    # childs of this node in the up tree
    childs = []
    # link to next node with same item Id (for the header table)
    nodeLink = -1
    # -1 means null
    parent = -1

    def __init__(self):
        self.itemId = -1
        self.count = 1
        self.nodeUtility = 0
        self.childs = []
        self.nodeLink = -1
        self.parent = -1

    def getChildWithId(self, name):
        flag = 0
        for child in self.childs:
            if child.itemId == name:
                flag = 1
                return child
        if flag == 0:
            # no child with given id
            return -1