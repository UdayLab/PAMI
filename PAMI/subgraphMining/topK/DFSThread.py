import threading

class DfsThread(threading.Thread):
    """
       A thread class for performing DFS-based subgraph mining on a set of candidates.

       Args:
           graphDb (GraphDatabase): The graph database containing the input graphs.
           candidates (Queue): A queue containing candidate subgraphs to be mined.
           minSup (int): The minimum support threshold for frequent subgraph mining.
           tkgInstance (TKGInstance): An instance of the Top-K Graphs (TKG) algorithm for dynamic subgraph mining.
       """

    def __init__(self, graphDb, candidates, minSup, tkgInstance):
        threading.Thread.__init__(self)
        self.graphDb = graphDb
        self.candidates = candidates
        self.minSup = minSup
        self.tkgInstance = tkgInstance

    def run(self):
        """
            Runs the DFS-based subgraph mining process on the candidate subgraphs.

            This method is invoked when the thread is started using the `start()` method.

            Within the mining loop:
            - Retrieves a candidate subgraph from the candidates queue.
            - Checks if the candidate's support meets the minimum support threshold.
            - If the support is sufficient, invokes the gspanDynamicDFS method of the TKGInstance
              to perform dynamic DFS-based subgraph mining with the candidate's DFS code,
              the graph database, and the set of graph IDs associated with the candidate.

            This method continues to run until the candidates queue is empty or until a candidate
            with insufficient support is encountered.
            """
        while not self.candidates.empty():
            _, candidate = self.candidates.get()
            if len(candidate.setOfGraphsIds) < self.minSup:
                break
            self.tkgInstance.gspanDynamicDFS(candidate.dfsCode, self.graphDb, candidate.setOfGraphsIds)
