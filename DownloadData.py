from graphein.protein.utils import download_alphafold_structure

#fp = download_alphafold_structure("B7MBF8", out_dir="/tmp", aligned_score=False)
#g = construct_graph(config=config, path=fp)

file = open("DeepTMHMM.3line", "r")
#while True:


while True:
    line = file.readline().strip()
    if line == '':
        break
    protein_name = line.split("|")[0][1:]
    
    file.readline()
    file.readline()
    download_alphafold_structure(protein_name, out_dir="data/", aligned_score=False)

file.close()