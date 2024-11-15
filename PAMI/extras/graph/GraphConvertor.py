#  Usage
#  obj1 = GraphConvertor('iFile', 'oFile')
#
#  obj1.convertTraditional2Compressed()
#
#  obj1.getMemoryRSS()
#
#  obj1.getMemoryUSS()
#
#  obj2 = GraphConvertor('iFileCompressed', 'oFileTrad')
#
#  obj2.convertCompressed2Traditional()
#
#  obj2.getMemoryRSS()
#
#  obj2.getMemoryUSS()

import os
import psutil


class GraphConvertor:
    def __init__(self, iFile):
        self.iFile = iFile
        self.convertedData = []

    def _writeGraphToFileCompressed(self, graph):
        node_str = ' '.join(f"{node} {label}" for node, label in sorted(graph['nodes'], key=lambda x: x[0]))
        edge_str = ' '.join(f"{u} {v} {label}" for u, v, label in graph['edges'])
        return f"{node_str} : {edge_str}\n"

    def _writeGraphToFileTraditional(self, graph, gId):
        traditional_lines = [f"t # {gId}\n"]
        for node, label in sorted(graph['nodes'], key=lambda x: x[0]):
            traditional_lines.append(f"v {node} {label}\n")
        for u, v, label in graph['edges']:
            traditional_lines.append(f"e {u} {v} {label}\n")
        return ''.join(traditional_lines)

    def convertTraditional2Compressed(self):
        graph = {}
        self.convertedData = []
        with open(self.iFile, 'r') as iFile:
            for line in iFile:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 't':
                    if graph:
                        compressedGraph = self._writeGraphToFileCompressed(graph)
                        self.convertedData.append(compressedGraph)
                    graph = {'nodes': [], 'edges': []}
                elif parts[0] == 'v':
                    graph['nodes'].append((int(parts[1]), parts[2]))
                elif parts[0] == 'e':
                    graph['edges'].append((int(parts[1]), int(parts[2]), parts[3]))
            if graph:
                compressedGraph = self._writeGraphToFileCompressed(graph)
                self.convertedData.append(compressedGraph)

    def convertCompressed2Traditional(self):
        self.convertedData = []
        gId = 0
        with open(self.iFile, 'r') as iFile:
            for line in iFile:
                if not line.strip():
                    continue  # Skip empty lines
                if ':' not in line:
                    print(f"Invalid format in line: {line.strip()}")
                    continue
                nodes_part, edges_part = line.strip().split(':')
                nodes_tokens = nodes_part.strip().split()
                edges_tokens = edges_part.strip().split()

                # Parse nodes
                nodes = []
                for i in range(0, len(nodes_tokens), 2):
                    node_id = int(nodes_tokens[i])
                    node_label = nodes_tokens[i + 1]
                    nodes.append((node_id, node_label))

                # Parse edges
                edges = []
                for i in range(0, len(edges_tokens), 3):
                    if i + 2 >= len(edges_tokens):
                        print(f"Incomplete edge information in line: {line.strip()}")
                        break
                    u = int(edges_tokens[i])
                    v = int(edges_tokens[i + 1])
                    label = edges_tokens[i + 2]
                    edges.append((u, v, label))

                graph = {'nodes': nodes, 'edges': edges}
                traditionalGraph = self._writeGraphToFileTraditional(graph, gId)
                self.convertedData.append(traditionalGraph)
                gId += 1

    def save(self, oFile):
        """
        Saves the converted data to the specified output file.

        :param oFile: Path to the output file.
        """
        if not self.convertedData:
            print("No converted data to save. Please perform a conversion first.")
            return

        with open(oFile, 'w') as file:
            for graphData in self.convertedData:
                file.write(graphData)

    def getMemoryRSS(self):
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        return rss

    def getMemoryUSS(self):
        process = psutil.Process(os.getpid())
        uss = process.memory_full_info().uss
        return uss

