import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import os
import community as community_louvain

# Load the data
def load_airports(file_path='airports.dat'):
    """Load airports data with named columns"""
    cols = ['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 
            'Altitude', 'Timezone', 'DST', 'Tz database time zone', 'Type', 'Source']
    
    airports = pd.read_csv(file_path, header=None, names=cols)
    # Filter invalid airports (missing coordinates or IATA code)
    airports = airports[~airports['Latitude'].isna() & ~airports['Longitude'].isna()]
    airports = airports[airports['IATA'] != '\\N']
    airports = airports[airports['Latitude'] != '\\N']
    airports = airports[airports['Longitude'] != '\\N']

    return airports

def load_routes(file_path='routes.dat'):
    """Load routes data with named columns"""
    cols = ['Airline', 'Airline ID', 'Source airport', 'Source airport ID',
            'Destination airport', 'Destination airport ID', 'Codeshare', 'Stops', 'Equipment']
    
    routes = pd.read_csv(file_path, header=None, names=cols)
    return routes

# Create the flight network
def create_flight_network(airports_df, routes_df):
    """Create a network graph from airports and routes data"""
    
    unique_route_airports = set(routes_df['Source airport'].unique()) | set(routes_df['Destination airport'].unique())
    
    # Filter airports to only those in routes
    airports_df = airports_df[airports_df['IATA'].isin(unique_route_airports)]
    
    G = nx.DiGraph()
    
    # Create a lookup dictionary for faster access
    airport_data = {}
    for _, row in airports_df.iterrows():
        iata = row['IATA']
        airport_data[iata] = {
            'name': row['Name'],
            'city': row['City'],
            'country': row['Country'],
            'latitude': row['Latitude'],
            'longitude': row['Longitude']
        }
    
    # Add nodes (airports)
    for iata, data in airport_data.items():
        G.add_node(iata, **data)
    
    # Count routes between airport pairs
    route_counts = {}
    for _, row in routes_df.iterrows():
        source = row['Source airport']
        dest = row['Destination airport']
        
        if source in G.nodes and dest in G.nodes:
            key = (source, dest)
            route_counts[key] = route_counts.get(key, 0) + 1
    
    # Add edges (routes) with weights
    for (source, dest), count in route_counts.items():
        G.add_edge(source, dest, weight=count)
    
    return G

# Calculate network statistics
def calculate_network_statistics(G):
    """Calculate basic network statistics"""
    stats = {}
    
    # Basic metrics
    stats['num_nodes'] = len(G.nodes())
    stats['num_edges'] = len(G.edges())
    stats['avg_degree'] = sum(dict(G.degree()).values()) / stats['num_nodes']
    stats['network_density'] = nx.density(G)
    
    # Degree distribution
    degrees = [d for _, d in G.degree()]
    stats['max_degree'] = max(degrees)
    stats['min_degree'] = min(degrees)
    stats['median_degree'] = np.median(degrees)
    
    # Connected components
    stats['num_weakly_connected'] = nx.number_weakly_connected_components(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    stats['largest_wcc_size'] = len(largest_wcc)
    stats['largest_wcc_percentage'] = stats['largest_wcc_size'] / stats['num_nodes'] * 100
    
    # Clustering coefficient (convert to undirected for this)
    G_undirected = G.to_undirected()
    stats['avg_clustering'] = nx.average_clustering(G_undirected)
    
    # Path lengths in largest component (can be slow for large networks)
    subgraph = G.subgraph(largest_wcc)
    try:
        stats['avg_path_length'] = nx.average_shortest_path_length(subgraph)
    except:
        stats['avg_path_length'] = "Too large to compute"
    
    try:
        if nx.is_weakly_connected(subgraph):
            stats['diameter'] = nx.diameter(subgraph)
        else:
            stats['diameter'] = "Network is not connected"
    except:
        stats['diameter'] = "Too large to compute"
    
    return stats

def plot_degree_distribution(G, filename="output/degree_distribution.png"):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    degrees = [G.degree(n) for n in G.nodes()]

    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=range(1, max(degrees)+2), edgecolor='black', alpha=0.7)
    plt.title("Degree Distribution (Total Degree in DiGraph)")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Degree distribution figure saved to {filename}")

# Calculate centrality measures
def calculate_centrality_metrics(G):
    """Calculate various centrality metrics for the network"""
    # Degree centrality
    degree_centrality = dict(G.degree())
    
    # Betweenness centrality (using approximation for large networks)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Eigenvector centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=3000)
    except:
        # Fallback if eigenvector centrality doesn't converge
        eigenvector_centrality = nx.degree_centrality(G)
        print("Warning: Eigenvector centrality calculation did not converge, using degree centrality as fallback")
    
    return degree_centrality, betweenness_centrality, eigenvector_centrality

# Get top airports by centrality measure
def get_top_airports(centrality_dict, n=10):
    """Get the top n airports by a centrality measure"""
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

# Conduct robustness analysis
def analyze_robustness(G, top_n=20):
    """
    Analyze network robustness by simulating removal of key hubs
    
    Parameters:
    G (networkx.DiGraph): The flight network graph
    top_n (int): Number of airports to remove in sequence
    
    Returns:
    pandas.DataFrame: Results of robustness analysis
    """
    import pandas as pd
    import networkx as nx
    
    # Get original network metrics
    original_size = len(G)
    original_largest_cc = len(max(nx.weakly_connected_components(G), key=len))
    
    # Get top hubs by degree
    hub_degrees = dict(G.degree())
    top_hubs = sorted(hub_degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Results storage
    results = []
    
    # Simulate removal of hubs one by one
    G_copy = G.copy()
    removed_hubs = []
    
    for i, (hub, degree) in enumerate(top_hubs):
        # Store info about the hub
        hub_info = {
            'rank': i+1,
            'hub': hub,
            'degree': degree,
            'name': G.nodes[hub].get('name', 'Unknown'),
            'city': G.nodes[hub].get('city', 'Unknown'),
            'country': G.nodes[hub].get('country', 'Unknown')
        }
        
        # Remove the hub
        G_copy.remove_node(hub)
        removed_hubs.append(hub)
        
        # Calculate new metrics
        components = list(nx.weakly_connected_components(G_copy))
        
        if components:
            largest_cc = max(components, key=len)
            hub_info['nodes_remaining'] = len(G_copy)
            hub_info['largest_cc_size'] = len(largest_cc)
            hub_info['largest_cc_percentage'] = len(largest_cc) / original_size * 100
            hub_info['num_components'] = len(components)
            
            # Skip diameter calculation as it can be time-consuming
            hub_info['diameter'] = -1
        else:
            hub_info['nodes_remaining'] = 0
            hub_info['largest_cc_size'] = 0
            hub_info['largest_cc_percentage'] = 0
            hub_info['num_components'] = 0
            hub_info['diameter'] = -1
        
        # Add to results
        results.append(hub_info)
        print(f"Processed hub {i+1}/{top_n}: {hub}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# Base visualization function
def create_world_map_base(figsize=(16, 8), projection=ccrs.PlateCarree(), title="Flight Network Visualization"):
    """Create a base world map for visualization"""
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    
    # Add map features
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='#d9f2fb')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#888888')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='-', edgecolor='#888888')
    
    # Set title
    plt.title(title, fontsize=15)
    
    return fig, ax

# Function to visualize the complete flight network
def visualize_full_network(G, filename="full_network.png"):
    """Create a visualization of the complete flight network with all airports and routes"""
    fig, ax = create_world_map_base(title="Global Flight Network - All Airports and Routes")
    
    # Plot all routes first - but limit density by edge weight
    # Here we'll take routes that have at least 2 flights (to reduce visual clutter)
    min_weight = 1  # Minimum weight to include (adjust as needed)
    
    # First collect all edges
    weighted_edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    # Sort by weight (most frequent routes first)
    weighted_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Plot routes with color based on weight
    for source, target, weight in weighted_edges:
        if weight >= min_weight:
            # Get coordinates
            src_lon = G.nodes[source]['longitude']
            src_lat = G.nodes[source]['latitude']
            dst_lon = G.nodes[target]['longitude']
            dst_lat = G.nodes[target]['latitude']
            
            # Calculate line width and alpha based on weight
            linewidth = 0.1 + 0.05 * np.log1p(weight)
            alpha = min(0.1 + 0.02 * np.log1p(weight), 0.5)  # Cap at 0.5
            
            # Plot the route
            ax.plot([src_lon, dst_lon], [src_lat, dst_lat],
                    transform=ccrs.Geodetic(),
                    color='red',
                    linewidth=linewidth,
                    alpha=alpha)
    
    # Get degree centrality for node sizing
    degree = dict(G.degree())
    
    # Plot all airports with very small blue dots
    for airport in G.nodes():
        lon = G.nodes[airport]['longitude']
        lat = G.nodes[airport]['latitude']
        
        # Scale node size by degree but keep them very small
        node_size = 0.5 + 0.2 * np.log1p(degree.get(airport, 1))
        
        ax.plot(lon, lat, 'o',
                transform=ccrs.PlateCarree(),
                markersize=node_size,  # Very small dots
                color='blue',
                alpha=0.7)
    
    # Add statistics text
    plt.figtext(0.02, 0.02, 
                f"Airports: {len(G.nodes())}, Routes: {len(G.edges())}\n"
                f"Complete network visualization",
                fontsize=8)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Full network visualization saved to {filename}")

# Function to visualize top airports by centrality
def visualize_top_centrality(G, centrality_dict, centrality_name, top_n=10, filename=None):
    """Visualize top airports by a specific centrality measure"""
    if filename is None:
        filename = f"top_{centrality_name}_centrality.png"
    
    fig, ax = create_world_map_base(title=f"Top {top_n} Airports by {centrality_name.title()} Centrality")
    
    # Get top airports
    top_airports = get_top_airports(centrality_dict, top_n)
    
    # First plot all routes between all top airports - make them more prominent
    for i, (source, _) in enumerate(top_airports):
        for j, (target, _) in enumerate(top_airports):
            if source != target and G.has_edge(source, target):
                # Get coordinates
                src_lon = G.nodes[source]['longitude']
                src_lat = G.nodes[source]['latitude']
                dst_lon = G.nodes[target]['longitude']
                dst_lat = G.nodes[target]['latitude']
                
                # Plot the route with brighter color and higher alpha
                ax.plot([src_lon, dst_lon], [src_lat, dst_lat],
                        transform=ccrs.Geodetic(),
                        color='#FF0000',  # Bright red
                        linewidth=1.0,
                        alpha=0.6)
    
    # Plot all airports as very small blue dots (background airports)
    for airport in G.nodes():
        if airport not in [a for a, _ in top_airports]:
            lon = G.nodes[airport]['longitude']
            lat = G.nodes[airport]['latitude']
            
            ax.plot(lon, lat, 'o',
                    transform=ccrs.PlateCarree(),
                    markersize=0.5,
                    color='#000088',
                    alpha=0.2)
    
    # Plot top airports with ranking - make them stand out more
    for rank, (airport, centrality_value) in enumerate(top_airports, 1):
        lon = G.nodes[airport]['longitude']
        lat = G.nodes[airport]['latitude']
        
        ax.plot(lon, lat, 'o',
                transform=ccrs.PlateCarree(),
                markersize=6,
                color='#0000FF',
                alpha=1.0)
        
        # Add rank label
        ax.text(lon + 1, lat + 1, str(rank),
                transform=ccrs.PlateCarree(),
                fontsize=9,
                ha='center', va='center',
                color='white',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='circle', pad=0.3))
    
    # Add a legend with airport info
    legend_text = "\n".join([
        f"{rank}. {airport} ({G.nodes[airport]['city']}, {G.nodes[airport]['country']}): {value:.4f}"
        for rank, (airport, value) in enumerate(top_airports, 1)
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
    ax.text(0.02, 0.98, legend_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=props)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top {centrality_name} centrality visualization saved to {filename}")

# Function to visualize network after removing a hub
def visualize_network_without_hub(G, hub_to_remove, filename=None):
    """Visualize the network after removing a specific hub airport"""
    if filename is None:
        filename = f"network_without_{hub_to_remove}.png"
    
    # Create a copy of the graph and remove the hub
    G_copy = G.copy()
    
    # Get hub information for the title
    hub_info = f"{hub_to_remove}"
    if hub_to_remove in G.nodes():
        hub_name = G.nodes[hub_to_remove]['name']
        hub_city = G.nodes[hub_to_remove]['city']
        hub_country = G.nodes[hub_to_remove]['country']
        hub_info = f"{hub_to_remove} ({hub_name}, {hub_city}, {hub_country})"
    
    # Remove the hub
    if hub_to_remove in G_copy:
        G_copy.remove_node(hub_to_remove)
    
    # Create visualization
    fig, ax = create_world_map_base(
        title=f"Flight Network After Removing {hub_info}")
    
    # Get degree centrality
    degrees = dict(G_copy.degree())
    
    # Plot all routes with weight-based visibility
    weighted_edges = [(u, v, G_copy[u][v]['weight']) for u, v in G_copy.edges()]
    weighted_edges.sort(key=lambda x: x[2], reverse=True)
    
    for source, target, weight in weighted_edges:
        src_lon = G_copy.nodes[source]['longitude']
        src_lat = G_copy.nodes[source]['latitude']
        dst_lon = G_copy.nodes[target]['longitude']
        dst_lat = G_copy.nodes[target]['latitude']
        
        linewidth = 0.1 + 0.05 * np.log1p(weight)
        alpha = min(0.1 + 0.02 * np.log1p(weight), 0.5)
        
        ax.plot([src_lon, dst_lon], [src_lat, dst_lat],
                transform=ccrs.Geodetic(),
                color='red',
                linewidth=linewidth,
                alpha=alpha)
    
    # Plot remaining airports
    for airport in G_copy.nodes():
        lon = G_copy.nodes[airport]['longitude']
        lat = G_copy.nodes[airport]['latitude']
        
        node_size = 0.5 + 0.2 * np.log1p(degrees.get(airport, 1))
        
        ax.plot(lon, lat, 'o',
                transform=ccrs.PlateCarree(),
                markersize=node_size,
                color='blue',
                alpha=0.7)
    
    # Mark removed hub location
    if hub_to_remove in G.nodes():
        hub_lon = G.nodes[hub_to_remove]['longitude']
        hub_lat = G.nodes[hub_to_remove]['latitude']
        
        ax.plot(hub_lon, hub_lat, 'X',
                transform=ccrs.PlateCarree(),
                markersize=10,
                markeredgewidth=2,
                color='black',
                alpha=1.0)
        
        ax.text(hub_lon, hub_lat + 2, f"Removed: {hub_to_remove}",
                transform=ccrs.PlateCarree(),
                fontsize=9,
                ha='center', va='bottom',
                color='black',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    original_airports = len(G.nodes())
    original_routes = len(G.edges())
    remaining_airports = len(G_copy.nodes())
    remaining_routes = len(G_copy.edges())
    
    stats_text = (
        f"Original network: {original_airports} airports, {original_routes} routes\n"
        f"After removal: {remaining_airports} airports, {remaining_routes} routes\n"
        f"Impact: {original_routes - remaining_routes} routes removed ({(original_routes - remaining_routes) / original_routes * 100:.1f}%)"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=8)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network without {hub_to_remove} visualization saved to {filename}")

# Function to detect and visualize communities
def detect_and_visualize_communities(G, filename="community_detection.png"):
    """Detect and visualize communities in the flight network using Louvain algorithm"""
    
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    
    # Apply Louvain algorithm
    print("Detecting communities...")
    partition = community_louvain.best_partition(G_undirected)
    
    # Count airports in each community
    community_sizes = {}
    for airport, community in partition.items():
        if community not in community_sizes:
            community_sizes[community] = 0
        community_sizes[community] += 1
    
    # Get the top communities by size
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    top_communities = sorted_communities[:10]  # Focus on top 10 communities
    
    # Map each airport to its community
    airport_communities = {}
    for airport, community in partition.items():
        airport_communities[airport] = community
    
    # Create visualization
    fig, ax = create_world_map_base(title="Flight Network Community Detection (Louvain Algorithm)")
    
    # Create a colormap for communities
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_communities)))
    community_colors = {comm: colors[i] for i, (comm, _) in enumerate(top_communities)}
    
    # Plot background airports
    background_airports = [airport for airport in G.nodes() 
                          if airport_communities.get(airport) not in [c for c, _ in top_communities]]
    
    for airport in background_airports:
        lon = G.nodes[airport]['longitude']
        lat = G.nodes[airport]['latitude']
        
        ax.plot(lon, lat, 'o',
                transform=ccrs.PlateCarree(),
                markersize=0.5,
                color='gray',
                alpha=0.2)
    
    # Plot airports by community
    for (community, size) in top_communities:
        comm_airports = [airport for airport in G.nodes() 
                         if airport_communities.get(airport) == community]
        
        for airport in comm_airports:
            lon = G.nodes[airport]['longitude']
            lat = G.nodes[airport]['latitude']
            degree = G.degree(airport)
            
            node_size = 0.5 + 0.3 * np.log1p(degree)
            
            ax.plot(lon, lat, 'o',
                    transform=ccrs.PlateCarree(),
                    markersize=node_size,
                    color=community_colors[community],
                    alpha=0.7)
    
    # Create legend for communities
    legend_elements = []
    for i, (community, size) in enumerate(top_communities):
        comm_airports = [airport for airport in G.nodes() 
                         if airport_communities.get(airport) == community]
        if comm_airports:
            most_central = max(comm_airports, key=lambda x: G.degree(x))
            country = G.nodes[most_central].get('country', 'Unknown')
            
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              markersize=10, markerfacecolor=community_colors[community],
                                              label=f"Comm {i+1}: {size} airports (centered in {country})"))
    
    ax.legend(handles=legend_elements, title="Communities", 
              loc='lower left', fontsize=8)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Community detection visualization saved to {filename}")

# Function to analyze hub accuracy based on real-world flight patterns
def visualize_hub_accuracy(hub_accuracy_df, filename="output/hub_accuracy.png"):
    """
    Visualize how well different centrality measures predict hub importance
    With improved chart layout and design
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1.5, 1])
    
    # Create custom color scheme
    colors = {
        'degree': '#3498db',       # Blue
        'betweenness': '#e67e22',  # Orange
        'eigenvector': '#2ecc71'   # Green
    }
    
    # ==== Chart 1: Difference heatmap - showing centrality differences by traffic rank ====
    ax1 = plt.subplot(gs[0, :])
    
    # Prepare heatmap data
    airports = hub_accuracy_df['Airport'].tolist()
    traffic_ranks = hub_accuracy_df['Traffic_Rank'].tolist()
    
    # Sort by traffic rank
    sorted_indices = np.argsort(traffic_ranks)
    sorted_airports = [airports[i] for i in sorted_indices]
    
    # Extract ranking difference data
    diff_data = []
    for i in sorted_indices:
        row = hub_accuracy_df.iloc[i]
        diff_data.append([
            row['Degree_Diff'] if not np.isnan(row['Degree_Diff']) else 0,
            row['Betweenness_Diff'] if not np.isnan(row['Betweenness_Diff']) else 0,
            row['Eigenvector_Diff'] if not np.isnan(row['Eigenvector_Diff']) else 0
        ])
    diff_data = np.array(diff_data)
    
    # Create heatmap
    im = ax1.imshow(diff_data.T, cmap='YlOrRd', aspect='auto')
    cbar = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.01)
    cbar.set_label('Ranking Difference', fontsize=10)
    
    # Set labels
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Degree Centrality', 'Betweenness Centrality', 'Eigenvector Centrality'])
    
    # Show names for top 10 airports, use traffic ranks for the rest
    x_labels = []
    for i, idx in enumerate(sorted_indices):
        airport = airports[idx]
        rank = traffic_ranks[idx]
        if i < 10:
            city = hub_accuracy_df.iloc[idx]['City']
            x_labels.append(f"{airport}\n({city}, #{rank})")
        else:
            x_labels.append(f"#{rank}")
    
    ax1.set_xticks(range(len(sorted_airports)))
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    
    ax1.set_title('Centrality Ranking Differences by Airport (Sorted by Traffic Rank)', fontsize=14)
    
    # ==== Chart 2: Separated scatter plots - correlation between each centrality and traffic rank ====
    # Create 3 subplots for each centrality type
    centrality_types = [
        ('Degree_Rank', 'Degree Centrality', colors['degree'], 'o'),
        ('Betweenness_Rank', 'Betweenness Centrality', colors['betweenness'], 's'),
        ('Eigenvector_Rank', 'Eigenvector Centrality', colors['eigenvector'], '^')
    ]
    
    for i, (col, label, color, marker) in enumerate(centrality_types):
        ax = plt.subplot(gs[1, i % 2] if i < 2 else gs[2, 0])
        
        # Get valid data points
        valid_data = hub_accuracy_df[~hub_accuracy_df[col].isna()]
        x = valid_data['Traffic_Rank']
        y = valid_data[col]
        
        # Calculate correlation coefficient
        correlation = valid_data['Traffic_Rank'].corr(valid_data[col])
        
        # Draw scatter plot
        ax.scatter(x, y, color=color, marker=marker, s=70, alpha=0.7, label=label)
        
        # Draw ideal line and actual trend line
        max_rank = max(max(x), max(y))
        ax.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.5, label='Perfect Correlation')
        
        # Calculate trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(x), p(np.sort(x)), color=color, linestyle='-', alpha=0.5)
        
        # Add correlation coefficient label
        ax.text(0.05, 0.95, f"Correlation: {correlation:.2f}", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Mark airports with largest differences
        diff_col = col.replace('_Rank', '_Diff')
        largest_diff = valid_data.nlargest(3, diff_col)
        
        for _, row in largest_diff.iterrows():
            ax.annotate(row['Airport'], 
                        xy=(row['Traffic_Rank'], row[col]),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=8, color='red', fontweight='bold')
        
        ax.set_xlabel('Traffic Rank', fontsize=10)
        ax.set_ylabel('Centrality Rank', fontsize=10)
        ax.set_title(f'Correlation between {label} and Traffic Rank', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=9)
        
        # Unify axis ranges
        ax.set_xlim(0, max_rank + 1)
        ax.set_ylim(0, max_rank + 1)
    
    # ==== Chart 3: Ranking difference distribution ====
    ax3 = plt.subplot(gs[2, 1])
    
    # Calculate mean differences and standard deviations
    mean_diffs = [
        hub_accuracy_df['Degree_Diff'].mean(),
        hub_accuracy_df['Betweenness_Diff'].mean(),
        hub_accuracy_df['Eigenvector_Diff'].mean()
    ]
    
    std_diffs = [
        hub_accuracy_df['Degree_Diff'].std(),
        hub_accuracy_df['Betweenness_Diff'].std(),
        hub_accuracy_df['Eigenvector_Diff'].std()
    ]
    
    x_pos = np.arange(len(mean_diffs))
    labels = ['Degree', 'Betweenness', 'Eigenvector']
    bar_colors = [colors['degree'], colors['betweenness'], colors['eigenvector']]
    
    # Draw bar chart with error bars
    bars = ax3.bar(x_pos, mean_diffs, yerr=std_diffs, capsize=5, 
                  color=bar_colors, alpha=0.7, width=0.6)
    
    # Add value labels on bars
    for bar, value, std in zip(bars, mean_diffs, std_diffs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, 
                f"{value:.2f}±{std:.2f}", ha='center', va='bottom', fontsize=9)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Average Ranking Difference', fontsize=10)
    ax3.set_title('Average Ranking Differences with Standard Deviations', fontsize=12)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle('Correlation Analysis Between Airport Centrality Metrics and Actual Traffic', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Improved hub accuracy visualization saved to {filename}")

def analyze_hub_accuracy(G, top_n=20):
    """Analyze how well centrality metrics predict real-world hub importance (improved version)"""
    # Calculate centrality measures
    degree_cent, betweenness_cent, eigenvector_cent = calculate_centrality_metrics(G)
    
    # Get top airports by each measure
    top_degree = get_top_airports(degree_cent, top_n)
    top_betweenness = get_top_airports(betweenness_cent, top_n)
    top_eigenvector = get_top_airports(eigenvector_cent, top_n)
    
    # Get unique set of airports from all centrality measures
    all_top_airports = set()
    for airport, _ in top_degree + top_betweenness + top_eigenvector:
        all_top_airports.add(airport)
    
    # Calculate total traffic (approx) by sum of in/out edge weights
    airport_traffic = {}
    for airport in all_top_airports:
        in_edges = G.in_edges(airport, data=True)
        in_traffic = sum(data.get('weight', 1) for _, _, data in in_edges)
        
        out_edges = G.out_edges(airport, data=True)
        out_traffic = sum(data.get('weight', 1) for _, _, data in out_edges)
        
        airport_traffic[airport] = in_traffic + out_traffic
    
    # Sort by traffic
    top_by_traffic = sorted(airport_traffic.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Set up more airports for a more comprehensive comparison
    extended_top_n = max(top_n, 30)  # Extend to at least 30 airports for comparison
    all_centrality_ranks = {}
    
    # Get extended rankings
    for i, (airport, _) in enumerate(get_top_airports(degree_cent, extended_top_n), 1):
        if airport not in all_centrality_ranks:
            all_centrality_ranks[airport] = {}
        all_centrality_ranks[airport]['Degree_Rank'] = i
    
    for i, (airport, _) in enumerate(get_top_airports(betweenness_cent, extended_top_n), 1):
        if airport not in all_centrality_ranks:
            all_centrality_ranks[airport] = {}
        all_centrality_ranks[airport]['Betweenness_Rank'] = i
    
    for i, (airport, _) in enumerate(get_top_airports(eigenvector_cent, extended_top_n), 1):
        if airport not in all_centrality_ranks:
            all_centrality_ranks[airport] = {}
        all_centrality_ranks[airport]['Eigenvector_Rank'] = i
    
    # Collect all data
    comparison_data = []
    for traffic_rank, (airport, traffic) in enumerate(top_by_traffic, 1):
        airport_data = {
            'Airport': airport,
            'Name': G.nodes[airport]['name'],
            'City': G.nodes[airport]['city'],
            'Country': G.nodes[airport]['country'],
            'Traffic': traffic,
            'Traffic_Rank': traffic_rank
        }
        
        # Add centrality rankings
        if airport in all_centrality_ranks:
            ranks = all_centrality_ranks[airport]
            for rank_type in ['Degree_Rank', 'Betweenness_Rank', 'Eigenvector_Rank']:
                airport_data[rank_type] = ranks.get(rank_type)
                
                # Calculate differences
                if airport_data[rank_type] is not None:
                    diff_type = rank_type.replace('_Rank', '_Diff')
                    airport_data[diff_type] = abs(traffic_rank - airport_data[rank_type])
        
        comparison_data.append(airport_data)
    
    # Convert to DataFrame
    hub_accuracy_df = pd.DataFrame(comparison_data)
    
    # Calculate related statistics
    for metric in ['Degree', 'Betweenness', 'Eigenvector']:
        diff_col = f'{metric}_Diff'
        if diff_col in hub_accuracy_df.columns:
            mean_diff = hub_accuracy_df[diff_col].mean()
            median_diff = hub_accuracy_df[diff_col].median()
            max_diff = hub_accuracy_df[diff_col].max()
            std_diff = hub_accuracy_df[diff_col].std()
            
            print(f"\n{metric} Centrality difference from traffic ranking:")
            print(f"  Mean difference: {mean_diff:.2f}")
            print(f"  Median difference: {median_diff:.2f}")
            print(f"  Maximum difference: {max_diff:.2f}")
            print(f"  Standard deviation: {std_diff:.2f}")
    
    # Use improved visualization
    visualize_hub_accuracy(hub_accuracy_df)
    return hub_accuracy_df

# Function to visualize robustness analysis results
def visualize_robustness_analysis(results_df, filename="output/robustness_analysis.png"):
    """
    Visualize network robustness analysis results
    
    Parameters:
    results_df (pandas.DataFrame): Results of robustness analysis
    filename (str): Output filename
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    plt.figure(figsize=(15, 10))
    
    gs = gridspec.GridSpec(2, 2)
    
    # Plot 1: Largest connected component size
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(range(1, len(results_df) + 1), results_df['largest_cc_percentage'], 'o-', color='blue', linewidth=2)
    ax1.set_xlabel('Number of Hubs Removed', fontsize=10)
    ax1.set_ylabel('Largest Connected Component (%)', fontsize=10)
    ax1.set_title('Network Fragmentation After Hub Removal', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Number of components
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(range(1, len(results_df) + 1), results_df['num_components'], 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Number of Hubs Removed', fontsize=10)
    ax2.set_ylabel('Number of Components', fontsize=10)
    ax2.set_title('Component Proliferation', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: table of removed hubs
    ax3 = plt.subplot(gs[1, :])
    
    table_data = []
    for i, row in results_df.head(10).iterrows():
        hub_info = f"{row['hub']} ({row['city']}, {row['country']})"
        cc_impact = f"{row['largest_cc_percentage']:.1f}%"
        
        table_data.append([
            f"{i+1}",
            hub_info,
            f"{row['degree']}",
            cc_impact,
            f"{row['num_components']}"
        ])
    
    table = ax3.table(
        cellText=table_data,
        colLabels=["Rank", "Hub Airport", "Degree", "Largest CC Size", "Components"],
        loc='center',
        cellLoc='center',
        colWidths=[0.05, 0.4, 0.1, 0.15, 0.1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax3.set_title('Impact of Hub Removal on Network Structure', fontsize=12)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Robustness analysis visualization saved to {filename}")

def run_robustness_analysis(G):
    """Run the simplified robustness analysis"""
    print("\nConducting robustness analysis...")
    
    # Analyze network resilience
    results_df = analyze_robustness(G, top_n=20)
    
    # Create standard visualization
    visualize_robustness_analysis(results_df)
    
    print("Robustness analysis completed!")
    return results_df

def main():
    print("Loading data...")
    airports_df = load_airports()
    routes_df = load_routes()

    print(f"Loaded {len(airports_df)} airports and {len(routes_df)} routes")

    print("\nCreating flight network...")
    G = create_flight_network(airports_df, routes_df)
    print(f"Network created with {len(G.nodes())} airports and {len(G.edges())} routes")
    
    print("\nCalculating network statistics...")
    stats = calculate_network_statistics(G)
    print("\nNetwork Statistics:")
    print(f"Number of airports: {stats['num_nodes']}")
    print(f"Number of routes: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Network density: {stats['network_density']:.6f}")
    print(f"Average clustering coefficient: {stats['avg_clustering']:.4f}")
    print(f"Weakly connected components: {stats['num_weakly_connected']}")
    print(f"Largest component size: {stats['largest_wcc_size']} ({stats['largest_wcc_percentage']:.2f}%)")
    
    plot_degree_distribution(G, filename="output/degree_distribution.png")

    print("\nCalculating centrality metrics...")
    degree_cent, betweenness_cent, eigenvector_cent = calculate_centrality_metrics(G)
    
    print("\nAnalyzing top airports by centrality...")
    top_10_degree = get_top_airports(degree_cent, 10)
    top_10_betweenness = get_top_airports(betweenness_cent, 10)
    top_10_eigenvector = get_top_airports(eigenvector_cent, 10)
    
    print("\nTop 10 airports by degree centrality:")
    for i, (airport, value) in enumerate(top_10_degree, 1):
        name = G.nodes[airport]['name']
        city = G.nodes[airport]['city']
        country = G.nodes[airport]['country']
        print(f"{i}. {airport} ({name}, {city}, {country}): {value}")
    
    print("\nTop 10 airports by betweenness centrality:")
    for i, (airport, value) in enumerate(top_10_betweenness, 1):
        name = G.nodes[airport]['name']
        city = G.nodes[airport]['city']
        country = G.nodes[airport]['country']
        print(f"{i}. {airport} ({name}, {city}, {country}): {value:.6f}")
    
    print("\nTop 10 airports by eigenvector centrality:")
    for i, (airport, value) in enumerate(top_10_eigenvector, 1):
        name = G.nodes[airport]['name']
        city = G.nodes[airport]['city']
        country = G.nodes[airport]['country']
        print(f"{i}. {airport} ({name}, {city}, {country}): {value:.6f}")
    
    print("\nAnalyzing hub accuracy based on real-world flight patterns...")
    hub_accuracy_df = analyze_hub_accuracy(G)

    print("\nSaving hub accuracy data...")
    hub_accuracy_df.to_csv("output/hub_accuracy_data.csv", index=False)
    print("Hub accuracy data saved to output/hub_accuracy_data.csv")

    print("\nConducting robustness analysis...")
    robustness_df = run_robustness_analysis(G)
    
    print("\nCreating visualizations...")
    os.makedirs("output", exist_ok=True)
    
    visualize_full_network(G, filename="output/full_network.png")
    
    visualize_top_centrality(G, degree_cent, "degree", filename="output/top_degree_centrality.png")
    visualize_top_centrality(G, betweenness_cent, "betweenness", filename="output/top_betweenness_centrality.png")
    visualize_top_centrality(G, eigenvector_cent, "eigenvector", filename="output/top_eigenvector_centrality.png")
        
    visualize_network_without_hub(G, top_10_degree[0][0], filename="output/network_without_top_hub.png")
    visualize_robustness_analysis(robustness_df, filename="output/robustness_analysis.png")

    
    # Community detection
    detect_and_visualize_communities(G, filename="output/community_detection.png")
    
    print("\nAll analyses completed and visualizations saved to the 'output' directory!")

if __name__ == "__main__":
    main()