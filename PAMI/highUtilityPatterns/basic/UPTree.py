from UPNode import UPNode


class UPTree:
    headerList = []
    hasMoreThanOnePath = False
    # List of pairs (item, utility) of the header table
    mapItemNodes = {}
    # root of this tree
    root = UPNode()
    # Map of the item to its last node while traversing
    mapItemToLastNode = {}

    def __init__(self):
        self.headerList = []
        self.hasMoreThanOnePath = False
        self.mapItemToLastNode = {}
        self.mapItemNodes = {}
        self.root = UPNode()

    def addTransaction(self, transaction, RTU):
        currentNode = self.root
        NumberOfNodes = 0
        RemainingUtility = 0
        for idx, item in enumerate(transaction):
            itemName = item.name
            child = currentNode.getChildWithId(itemName)
            RemainingUtility += item.utility
            if child == -1:
                NumberOfNodes += 1
                nodeUtility = RemainingUtility
                # there is no node so we have to create new node
                currentNode = self.insertNewNode(currentNode, itemName, nodeUtility)
            else:
                # there is a node already we update it
                child.count += 1
                child.nodeUtility += RemainingUtility
                currentNode = child
        return NumberOfNodes

    def addLocalTransaction(self, localPath, pathUtility, mapItemToMinimumItemutility, pathCount):
        currentLocalNode = self.root
        RemainingUtility = 0
        NumberOfNodes = 0
        for item in localPath:
            RemainingUtility += mapItemToMinimumItemutility[item] * pathCount
        for item in localPath:
            RemainingUtility -= mapItemToMinimumItemutility[item] * pathCount
            child = currentLocalNode.getChildWithId(item)
            if child == -1:
                # if no child exists create a new node
                NumberOfNodes += 1
                currentLocalNode = self.insertNewNode(currentLocalNode, item, pathUtility - RemainingUtility)
            else:
                # child exists then update its count and utility
                child.count += 1
                child.nodeUtility += (pathUtility - RemainingUtility)
                currentLocalNode = child
        return NumberOfNodes

    def insertNewNode(self, currentlocalNode, itemName, nodeUtility):
        # create new node
        newNode = UPNode()
        newNode.itemId = itemName
        newNode.count = 1
        newNode.nodeUtility = nodeUtility
        newNode.parent = currentlocalNode
        # we link the new node to its parent
        currentlocalNode.childs.append(newNode)
        # check if tree has more than one path
        if not self.hasMoreThanOnePath and len(currentlocalNode.childs) > 1:
            self.hasMoreThanOnePath = True
        # we update the header table
        if itemName in self.mapItemNodes:
            # if already exist in table then we find the last node
            lastNode = self.mapItemToLastNode[itemName]
            # we add new node to nodelink of the last node
            lastNode.nodeLink = newNode
            # finally we set this new node as the last node
            self.mapItemToLastNode[itemName] = newNode
        else:
            # item deosnt exist in header table means first time appearing in table
            self.mapItemNodes[itemName] = newNode
            self.mapItemToLastNode[itemName] = newNode
        return newNode

    def createHeaderList(self, mapItemToTwu):
        self.headerList = list(self.mapItemNodes.keys())
        # print('map itemt top nodes')
        # print(self.mapItemNodes.keys())
        # print('map item to path utility')
        # print(mapItemToTwu)
        # sort the header list in decreasing order of twu
        self.headerList = sorted(self.headerList, key=lambda x: mapItemToTwu[x], reverse=True)