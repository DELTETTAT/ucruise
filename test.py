# import networkx as nx
# import matplotlib.pyplot as plt

# # === Load the GraphML file ===
# graphml_path = r"C:\Users\pc\Desktop\projects\driver_and_routes\clustering_cab_api replit\tricity_main_roads.graphml"
# G = nx.read_graphml(graphml_path)

# # === Basic Graph Info ===
# print("✅ Graph Loaded")
# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())

# # === Sample Node Attributes ===
# print("\n🔍 Sample Node Attributes:")
# for i, (node, data) in enumerate(G.nodes(data=True)):
#     if i >= 10: break
#     print(f"{node}: {data}")

# # === Sample Edge Attributes ===
# print("\n🔍 Sample Edge Attributes:")
# for i, (u, v, data) in enumerate(G.edges(data=True)):
#     if i >= 10: break
#     print(f"{u} -> {v}: {data}")

# # === Subgraph Sampling for Visualization ===
# sample_size = 500  # Adjust based on your system's memory
# sample_nodes = list(G.nodes())[:sample_size]
# H = G.subgraph(sample_nodes)

# # === Layout Optimization ===
# pos = nx.kamada_kawai_layout(H)  # Faster and cleaner for large graphs

# # === Draw the Subgraph ===
# plt.figure(figsize=(12, 8))
# nx.draw(
#     H, pos,
#     with_labels=False,
#     node_color='skyblue',
#     edge_color='gray',
#     node_size=30,
#     width=0.5
# )
# plt.title("🚗 Tricity Road Network (Sampled Subgraph)")
# plt.tight_layout()
# plt.show()


import networkx as nx
import folium

# === Load the GraphML file ===
graphml_path = r"C:\Users\pc\Desktop\projects\driver_and_routes\clustering_cab_api replit\tricity_main_roads.graphml"
G = nx.read_graphml(graphml_path)

# === Extract Coordinates ===
# Adjust keys if your GraphML uses 'x'/'y' instead of 'lat'/'lon'
latitudes = []
longitudes = []
for node, data in G.nodes(data=True):
    try:
        lat = float(data.get('lat') or data.get('y'))
        lon = float(data.get('lon') or data.get('x'))
        latitudes.append(lat)
        longitudes.append(lon)
    except (TypeError, ValueError):
        continue  # Skip nodes without valid coordinates

# === Center the Map ===
if latitudes and longitudes:
    center = [sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)]
else:
    raise ValueError("No valid coordinates found in the GraphML file.")

# === Create Folium Map ===
m = folium.Map(location=center, zoom_start=13, tiles='CartoDB positron')

# === Add Nodes as Circle Markers ===
for node, data in G.nodes(data=True):
    try:
        lat = float(data.get('lat') or data.get('y'))
        lon = float(data.get('lon') or data.get('x'))
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=str(node)
        ).add_to(m)
    except (TypeError, ValueError):
        continue

# === Add Edges as Lines (Optional) ===
for u, v in G.edges():
    try:
        lat_u = float(G.nodes[u].get('lat') or G.nodes[u].get('y'))
        lon_u = float(G.nodes[u].get('lon') or G.nodes[u].get('x'))
        lat_v = float(G.nodes[v].get('lat') or G.nodes[v].get('y'))
        lon_v = float(G.nodes[v].get('lon') or G.nodes[v].get('x'))
        folium.PolyLine(
            locations=[[lat_u, lon_u], [lat_v, lon_v]],
            color='gray',
            weight=1,
            opacity=0.5
        ).add_to(m)
    except (TypeError, ValueError, KeyError):
        continue

# === Save Map to HTML ===
m.save("tricity_road_network_map.html")
print("✅ Map saved as 'tricity_road_network_map.html'")

