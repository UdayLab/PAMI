#  Usage
#  obj = ConvertFormat('iFile', 'oFile')
#
import os
import psutil


class ConvertFormat:
    def __init__(self, iFile):
        self.iFile = iFile
        self.oFile = 'oFile.txt'

    def _writeGraphToFile(self, graph, oFile):
        node_str = ' '.join(f"{node} {label}" for node, label in sorted(graph['nodes']))
        edge_str = ' '.join(f"{u} {v} {label}" for u, v, label in graph['edges'])
        oFile.write(f"{node_str} : {edge_str}\n")

    def convert(self):
        graph = {}
        with open(self.iFile, 'r') as iFile, open(self.oFile, 'w') as oFile:
            for line in iFile:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 't':
                    if graph:
                        self._writeGraphToFile(graph, oFile)
                    graph = {'nodes': [], 'edges': []}
                elif parts[0] == 'v':
                    graph['nodes'].append((int(parts[1]), parts[2]))
                elif parts[0] == 'e':
                    graph['edges'].append((int(parts[1]), int(parts[2]), parts[3]))
            if graph:
                self._writeGraphToFile(graph, oFile)

    def getMemoryRSS(self):
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        return rss

    def getMemoryUSS(self):
        process = psutil.Process(os.getpid())
        uss = process.memory_full_info().uss
        return uss

