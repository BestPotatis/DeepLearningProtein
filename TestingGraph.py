from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
import matplotlib.pyplot as plt
import networkx as nx

config = ProteinGraphConfig()

# Constructing the graph from the protein file
proteinName = "Q99279"
g = construct_graph(config=config, path="data/"+proteinName+".pdb")

# The graph is a nx Graph
# https://networkx.org/documentation/stable/tutorial.html

# Get nodes and edges
print(g.nodes)
print(g.edges)
print("number of nodes: ", g.number_of_nodes(), \
      "\t number of edges: ", g.number_of_edges())

# overview of entire graph (type: dict)
print(g.graph)

# get individual dataframes from the graph (type: pd dataframe)
print(g.graph["pdb_df"]) # includes only alpha carbon info from each amino in the sequence
print(g.graph["raw_pdb_df"]) # includes all elements/atoms for each amino
print(g.graph["rgroup_df"]) # includes only r-group (side chain) information

print(g.graph["sequence_A"]) # idk what this one means.. it doesn't correspond to the sequence of the individual amino acids
print(g.graph["coords"]) # coords for all alpha-carbon amino acids in the protein

# get all individual node data including residue (amino acid) name or 3-D coordinates
print(g.nodes.data())
print(g.nodes.data("residue_name"))
print(g.nodes.data("coords")) # alpha carbon coords for each amino

# Visualize the graph
# https://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html
plot = plotly_protein_structure_graph(g, node_size_multiplier=0.2, label_node_ids=False, colour_nodes_by="residue_name")
plot.show()

# Graph analysis
# https://graphein.ai/notebooks/protein_graph_analytics.html
# https://graphein.ai/notebooks/dataloader_tutorial.html