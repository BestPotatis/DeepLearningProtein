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
#print(g.nodes)
#print(g.edges)

# Visualize the graph
# https://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html
plot = plotly_protein_structure_graph(g, node_size_multiplier=0.2, label_node_ids=False, colour_nodes_by="residue_name")
plot.show()

# Graph analysis
# https://graphein.ai/notebooks/protein_graph_analytics.html