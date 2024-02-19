from algo_gspan import Gspan

def main():
    input_file_path = 'chemical_340.txt'
    output_file_path = 'temp.txt'
    sup = [0.15, 0.2, 0.25, 0.3]
    runtime_340 = []
    for min_support in  sup:
        gspan_instance = Gspan()
        output_single_vertices = True  
        max_number_of_edges = 100
        gspan_instance.run(input_file_path, output_file_path, min_support, 
                                output_single_vertices, max_number_of_edges, 
                                    True)
        runtime_340.append(gspan_instance.runtime)


    print(runtime_340)

    input_file_path = 'chemical_422.txt'
    runtime_422 = []
    for min_support in sup:
        gspan_instance = Gspan()
        output_single_vertices = True  
        max_number_of_edges = 100
        gspan_instance.run(input_file_path, output_file_path, min_support, 
                                output_single_vertices, max_number_of_edges, 
                                    True)
        runtime_422.append(gspan_instance.runtime)
        
    print(runtime_422)

    input_file_path = 'mutag.txt'
    runtime_mut = []
    for min_support in sup:
        gspan_instance = Gspan()
        output_single_vertices = True  
        max_number_of_edges = 100
        gspan_instance.run(input_file_path, output_file_path, min_support, 
                                output_single_vertices, max_number_of_edges, 
                                    True)
        runtime_mut.append(gspan_instance.runtime)
        
    print(runtime_mut)


if __name__=='__main__':
    main()