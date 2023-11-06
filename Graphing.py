from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph

config = ProteinGraphConfig()

# Constructing the graph from the protein file
proteinName = "Q99279"
g = construct_graph(config=config, path="data/"+proteinName+".pdb")

# The graph is a nx Graph
# https://networkx.org/documentation/stable/tutorial.html

# Get nodes and edges
#g.nodes
#g.edges 

# Visualize the graph
# https://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html
plotly_protein_structure_graph(g, node_size_multiplier=0.5, colour_nodes_by="residue_name")

# Graph analysis
# https://graphein.ai/notebooks/protein_graph_analytics.html