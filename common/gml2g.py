import re


def resolve_edge_structure(es):
    result = []
    for element in es:
        arr = element.split()
        result.append([arr[2], arr[4], arr[6]])
    return result


def gml2g(file_name):
    node_pattern = re.compile(r'node (\[[^\]]*\])')
    edge_pattern = re.compile(r'edge (\[[^\]]*\])')

    with open(file_name) as f:
        text = f.read()

    node_structures = node_pattern.findall(text)
    edge_structures = edge_pattern.findall(text)

    file_name_sect = file_name.split(".")
    new_file_name = ".".join(file_name_sect[:-1]) + ".g"
    vnum, enum = len(node_structures), len(edge_structures)
    edges = resolve_edge_structure(edge_structures)

    with open(new_file_name, 'w') as f:
        f.write(str(vnum) + " " + str(enum) + "\n")
        for edge in edges:
            f.write(" ".join(edge))
            f.write("\n")


for i in range(97, 116):
    fn = "../datasets/H/H" + str(i) + ".gml"
    gml2g(fn)
