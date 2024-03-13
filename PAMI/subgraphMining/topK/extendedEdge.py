class ExtendedEdge:
    def __init__(self, v1, v2, vLabel1, vLabel2, edgeLabel):
        self.v1 = v1
        self.v2 = v2
        self.vLabel1 = vLabel1
        self.vLabel2 = vLabel2
        self.edgeLabel = edgeLabel
        self.hashCode = (1 + v1) * 100 + (1 + v2) * 50 + (1 + vLabel1) * 30 + (1 + vLabel2) * 20 + (1 + edgeLabel)

    def smallerThan(self, that):
        if that is None:
            return True

        x1, x2, y1, y2 = self.v1, self.v2, that.v1, that.v2

        if self.pairSmallerThan(x1, x2, y1, y2):
            return True
        elif x1 == y1 and x2 == y2:
            return (self.vLabel1 < that.vLabel1 or
                    (self.vLabel1 == that.vLabel1 and self.vLabel2 < that.vLabel2) or
                    (self.vLabel1 == that.vLabel1 and self.vLabel2 == that.vLabel2 and
                     self.edgeLabel < that.edgeLabel))
        else:
            return False

    def smallerThanOriginal(self, that):
        if that is None:
            return True

        x1, x2, y1, y2 = self.v1, self.v2, that.v1, that.v2

        if self.pairSmallerThan(x1, x2, y1, y2):
            return True
        elif x1 == y1 and x2 == y2:
            return (self.vLabel1 < that.vLabel1 or
                    (self.vLabel1 == that.vLabel1 and self.edgeLabel < that.edgeLabel) or
                    (self.vLabel1 == that.vLabel1 and self.edgeLabel == that.edgeLabel and
                     self.vLabel2 < that.vLabel2))
        else:
            return False

    def pairSmallerThan(self, x1, x2, y1, y2):
        xForward = x1 < x2
        yForward = y1 < y2

        if xForward and yForward:
            return x2 < y2 or (x2 == y2 and x1 > y1)
        elif not xForward and not yForward:
            return x1 < y1 or (x1 == y1 and x2 < y2)
        elif xForward:
            return x2 <= y1
        else:
            return x1 < y2

    def __hash__(self):
        return self.hashCode

    def __eq__(self, other):
        if not isinstance(other, ExtendedEdge):
            return False
        return (self.v1 == other.v1 and self.v2 == other.v2 and
                self.vLabel1 == other.vLabel1 and self.vLabel2 == other.vLabel2 and
                self.edgeLabel == other.edgeLabel)

    def __repr__(self):
        return f"<{self.v1},{self.v2},{self.vLabel1},{self.vLabel2},{self.edgeLabel}>"

    def getV1(self):
        return self.v1
    
    def getV2(self):
        return self.v2
    
    def getVLabel1(self):
        return self.vLabel1
    
    def getVLabel2(self):
        return self.vLabel2
    
    def getEdgeLabel(self):
        return self.edgeLabel
