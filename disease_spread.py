#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
disease_spread.py

A standalone example showing how to:
  1. Load airport & route data,
  2. Build a flight-network graph,
  3. Run a simplified SIR disease-spread simulation,
  4. Plot the S/I/R results over time.

Run it with:
    python disease_spread.py
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------------------------------------------------
# 1. Functions to load data & build the network (from your original code)
# -----------------------------------------------------------------------

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
    

def create_flight_network(airports_df, routes_df):
    """
    Create a directed flight-network graph from airports and routes data.
    Each edge has a 'weight' indicating how many routes connect that pair.
    """

    unique_route_airports = set(routes_df['Source airport'].unique()) | set(routes_df['Destination airport'].unique())
    
    # Filter airports to only those in routes
    airports_df = airports_df[airports_df['IATA'].isin(unique_route_airports)]
    
    G = nx.DiGraph()

    # Build a dict for fast airport lookup
    airport_data = {}
    for _, row in airports_df.iterrows():
        iata = row['IATA']
        airport_data[iata] = {
            'name': row['Name'],
            'city': row['City'],
            'country': row['Country'],
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
        }

    # Add airport nodes
    for iata, data in airport_data.items():
        G.add_node(iata, **data)

    # Count routes between pairs
    route_counts = {}
    for _, row in routes_df.iterrows():
        source = row['Source airport']
        dest = row['Destination airport']
        if source in G and dest in G:  # both must be valid nodes in G
            key = (source, dest)
            route_counts[key] = route_counts.get(key, 0) + 1

    # Add edges with 'weight'
    for (source, dest), count in route_counts.items():
        G.add_edge(source, dest, weight=count)

    return G

# -----------------------------------------------------------------------
# 2. SIR model functions
# -----------------------------------------------------------------------

def initialize_states(G, initially_infected=None):
    """
    Initialize each node's state: S (susceptible), I (infected), R (recovered/immune).
    """
    states = {}
    for node in G.nodes():
        states[node] = 'S'  # default to susceptible

    if initially_infected:
        for inf in initially_infected:
            if inf in states:
                states[inf] = 'I'  # mark as infected
    return states

def step_sir_model(G, states, beta=0.02, gamma=0.01):
    """
    Perform one discrete timestep of the SIR model:
      - beta: Probability an infected node transmits the disease to each neighbor.
      - gamma: Probability an infected node recovers (I->R) per step.
    """
    new_states = states.copy()

    # 1) Recovery step: I -> R with probability gamma
    for node in G.nodes():
        if states[node] == 'I':
            if random.random() < gamma:
                new_states[node] = 'R'

    # 2) Infection step: S -> I with probability = 1 - (1 - beta)^(#infected_neighbors)
    for node in G.nodes():
        if states[node] == 'S':
            infected_neighbors = 0
            for neighbor in G.neighbors(node):
                if states[neighbor] == 'I':
                    infected_neighbors += 1
            if infected_neighbors > 0:
                p_infection = 1.0 - (1.0 - beta)**infected_neighbors
                if random.random() < p_infection:
                    new_states[node] = 'I'

    return new_states

def simulate_sir(G, initially_infected, steps=50, beta=0.02, gamma=0.01):
    """
    Run a discrete SIR simulation on the flight network.
    Returns lists of S, I, R counts over time.
    """
    states = initialize_states(G, initially_infected)
    
    list_S = [sum(s == 'S' for s in states.values())]
    list_I = [sum(s == 'I' for s in states.values())]
    list_R = [sum(s == 'R' for s in states.values())]

    for t in range(1, steps+1):
        states = step_sir_model(G, states, beta=beta, gamma=gamma)
        list_S.append(sum(s == 'S' for s in states.values()))
        list_I.append(sum(s == 'I' for s in states.values()))
        list_R.append(sum(s == 'R' for s in states.values()))

    return list_S, list_I, list_R

# -----------------------------------------------------------------------
# 3. Visualization function for SIR results
# -----------------------------------------------------------------------

def plot_sir_curves(list_S, list_I, list_R, title="SIR Disease Spread Simulation",
                    out_filename="sir_simulation.png"):
    """
    Plot the evolution of Susceptible, Infected, and Recovered nodes over time.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(list_S, label='Susceptible', color='blue')
    plt.plot(list_I, label='Infected', color='red')
    plt.plot(list_R, label='Recovered', color='green')
    plt.xlabel("Time Step")
    plt.ylabel("Number of Airports")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save figure as PNG
    plt.savefig(out_filename, dpi=300)
    plt.close()
    print(f"Figure saved to {out_filename}")

# -----------------------------------------------------------------------
# 4. Main driver
# -----------------------------------------------------------------------

def main():
    print("Loading airports and routes...")
    airports_df = load_airports() 
    routes_df = load_routes()  
    G = create_flight_network(airports_df, routes_df)
    print(f"Loaded a network of {len(G.nodes())} airports and {len(G.edges())} routes.")

    # Example: Infect two major hubs initially
    initially_infected = ["FRA", "LAX"]  # Frankfurt, Los Angeles
    
    # Simulation parameters
    steps = 30
    beta = 0.02   # infection probability
    gamma = 0.01  # recovery probability

    print(f"\nStarting SIR simulation with initially_infected={initially_infected}, "
          f"steps={steps}, beta={beta}, gamma={gamma}")
    
    list_S, list_I, list_R = simulate_sir(G, initially_infected, steps=steps,
                                          beta=beta, gamma=gamma)

    # Print final result
    print(f"\nAt step {steps}:")
    print(f"  Susceptible: {list_S[-1]}")
    print(f"  Infected:    {list_I[-1]}")
    print(f"  Recovered:   {list_R[-1]}")

    # Plot the SIR curves
    output_filename = "SIR_Simulation_Result.png"
    plot_sir_curves(list_S, list_I, list_R,
                    title=f"SIR Model on Flight Network (Init: {initially_infected}, "
                          f"β={beta}, γ={gamma})")

if __name__ == "__main__":
    main()