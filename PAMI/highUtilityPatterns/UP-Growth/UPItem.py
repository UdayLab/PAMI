class UPItem:
    # the name of item
    name = 0
    # utility of the item
    utility = 0

    def __init__(self, name, utility):
        self.name = name
        self.utility = utility

    def getUtility(self):
        return self.utility

    def setUtility(self, utility):
        self.utility = utility

    def getName(self):
        return self.name