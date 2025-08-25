# import osmnx as ox

# # Replace with your target location
# place_name = "Sahibzada Ajit Singh Nagar, Punjab, India"


# # Download road network
# G = ox.graph_from_place(place_name, network_type='drive')

# # Plot it
# ox.plot_graph(G)



# try:
#     import osmnx as ox
#     from shapely.geometry import Point
#     HAS_OSMNX = True
#     print("✅ OSMnx loaded:", ox.__version__)
# except Exception as e:
#     HAS_OSMNX = False
#     import traceback
#     print("⚠️ OSMnx import FAILED; falling back to PCA. Traceback:")
#     traceback.print_exc()


import osmnx as ox
import osmnx as ox
import networkx as nx

places = [
    "Chandigarh, India",
    "Kharar, Punjab, India",
    "Sahibzada Ajit Singh Nagar district, Punjab, India"  # Mohali
]

custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]'

graphs = []
for p in places:
    try:
        Gp = ox.graph_from_place(p, custom_filter=custom_filter, simplify=True)
        graphs.append(Gp)
    except Exception as e:
        print(f"⚠️ Failed {p}: {e}")

# Merge them into one
G = nx.compose_all(graphs)

ox.save_graphml(G, "tricity_main_roads.graphml")
