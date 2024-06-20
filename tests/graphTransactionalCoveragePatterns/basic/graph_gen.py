import random
def generate_random_graph(n, m, max_label,f):
    vertices = [(f"v {i}", random.randint(1, max_label)) for i in range(n)]
    edges = set()

    while len(edges) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        label = random.randint(1, max_label)

        if u != v and (u, v, label) not in edges:
            edges.add((u, v, label))

    for vertex in vertices:
        s=f"{vertex[0]} {vertex[1]}\n"
        f.write(s)

    for edge in edges:
        s=f"e {edge[0]} {edge[1]} {edge[2]}\n"
        f.write(s)


def generate(path, transactions):
    f=open(path,"w+")
    for t in range(transactions):
        f.write("t # {}\n".format(t))
        generate_random_graph(random.randint(10,20),random.randint(20,30),3,f)

    f.write("t # -1")
    f.close()
