from PAMI.subgraphMining.basic.gspan import Gspan as alg


def main():
    input_file_path = 'chemical_340.txt'
    output_file_path = 'temp.txt'
    sup = [0.15, 0.2, 0.25, 0.3]
    runtime_340 = []
    for min_support in  sup:
        output_single_vertices = True  
        max_number_of_edges = 100
        obj = alg(input_file_path, output_file_path, min_support, 
                                output_single_vertices, max_number_of_edges, 
                                    True)

        obj.run()
        runtime_340.append(obj.getRuntime())


    print(runtime_340)



if __name__=='__main__':
    main()