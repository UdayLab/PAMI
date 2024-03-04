from PAMI.subgraphMining.basic import gspan as alg


def main():
    input_file_path = 'chemical_340.txt'
    output_file_path = 'temp.txt'
    sup = [0.15, 0.2, 0.25, 0.3]
    memRss = []
    memUSS = []
    runtime_340 = []
    for min_support in  sup:
        output_single_vertices = True  
        max_number_of_edges = 100
        obj = alg.GSpan(input_file_path, min_support, 
                                output_single_vertices, max_number_of_edges, 
                                    True)

        obj.run()
        obj.save('temp.txt')
        runtime_340.append(obj.getRuntime())
        memRss.append(obj.getMemoryRSS())
        memUSS.append(obj.getMemoryUSS())


    print(runtime_340)
    print(memRss)
    print(memUSS)

if __name__=='__main__':
    main()