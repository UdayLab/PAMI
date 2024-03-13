import threading

class DfsThread(threading.Thread):
    def __init__(self, graphDb, candidates, minSup, tkgInstance):
        threading.Thread.__init__(self)
        self.graphDb = graphDb
        self.candidates = candidates
        self.minSup = minSup
        self.tkgInstance = tkgInstance

    def run(self):
        while not self.candidates.empty():
            _, candidate = self.candidates.get()
            if len(candidate.setOfGraphsIds) < self.minSup:
                break
            self.tkgInstance.gspanDynamicDFS(candidate.dfsCode, self.graphDb, candidate.setOfGraphsIds)
