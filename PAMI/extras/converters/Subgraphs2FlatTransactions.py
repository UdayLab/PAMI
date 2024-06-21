# The goal of the below is to obtain flat transactions from the output of subgraph mining
# Flat transactions are transactions that contain the list of subgraphs that are present in a graph
#
#   from PAMI.extras.graph import flatTransactions as ft
#
#   obj = ft.FlatTransactions()
#
#   flatTransactions = obj.getFlatTransactions(fidGidDictMap)
#   (fidGidDictMap is a list of dictionaries with keys 'FID' and 'GIDs'
#   FID is subgraph/fragment ID and GIDs are the graph IDs that contain the subgraph)
#
#   obj.saveFlatTransactions(oFile)

class Subgraphs2FlatTransactions:

    def __init__(self):
        self.flatTransactions = {}

    def getFlatTransactions(self, fidGidDictMap):
        """
        fidGidMap is a list of dictionaries with keys 'FID' and 'GIDs'
        An example of this type of output is: getSubgraphGraphMapping in GSpan class
        from subgraphMining/basic/gspan.py in PAMI
        """
        graphToSubgraphs = {}

        for mapping in fidGidDictMap:
            fid = mapping['FID']
            gids = mapping['GIDs']

            for gid in gids:
                if gid not in graphToSubgraphs:
                    graphToSubgraphs[gid] = []
                graphToSubgraphs[gid].append(fid)

        for gid in graphToSubgraphs:
            graphToSubgraphs[gid] = sorted(set(graphToSubgraphs[gid]))

        self.flatTransactions = graphToSubgraphs
        return self.flatTransactions

    def saveFlatTransactions(self, oFile):
        """
        Save the available flat transactions to a file
        """
        with open(oFile, 'w') as f:
            for _, fids in self.flatTransactions.items():
                f.write(f"{' '.join(map(str, fids))}\n")
