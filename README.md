## Flight_Network

This repository contains a comprehensive analysis of the global flight network, including data loading, network construction, centrality computation, community detection, robustness tests, and a simplified SIR disease-spread simulation.

### Contents

1. **`flight_network.py`**  
   - Loads airport and route data from the OpenFlight dataset.  
   - Constructs a directed, weighted graph (using NetworkX).  
   - Performs various analyses:
     - **Degree, Betweenness, Eigenvector** centrality calculations.
     - **Community detection** (Louvain algorithm).
     - **Robustness** analysis by removing top-degree hubs.
     - **Visualization** of node/edge distributions, top hubs, and robustness results.
   
2. **`disease_spread.py`**  
   - Demonstrates a simplified **SIR model** on the constructed flight network.
   - Infects a set of initially chosen airports and simulates disease spread based on:
     - `beta` (infection probability)
     - `gamma` (recovery probability)
   - Plots time evolution of Susceptible, Infected, and Recovered node counts.

3. **`data`** (not included in this repo by default)  
   - Requires `airports.dat` and `routes.dat` from [OpenFlight](https://openflights.org/data.html).  
   - Make sure these files are placed in the same directory or adjust paths in the scripts.

4. **`output`**  
   - Where final plots and CSV files (e.g., `hub_accuracy_data.csv`, various PNG visuals) are saved.

5. **`README.md`** (this file)  
   - Overview of the repository structure and instructions on how to run the code.

### Prerequisites

- **Python 3** (tested on 3.8+)
- [**NetworkX**](https://networkx.org/)
- [**pandas**](https://pandas.pydata.org/)
- [**matplotlib**](https://matplotlib.org/)
- [**community-louvain**](https://pypi.org/project/python-louvain/) (for Louvain-based community detection)
- [**cartopy**](https://scitools.org.uk/cartopy/) (for geographic visualizations)

### Quick Start

1. **Clone** or download this repository:
   ```bash
   git clone https://github.com/Jessjx6/Flight_Network.git
   cd Flight_Network
   ```
2. **Install required packages**, for example using `conda`:
   ```bash
   conda install -c conda-forge networkx pandas matplotlib cartopy python-louvain
   ```
   or `pip`:
   ```bash
   pip install networkx pandas matplotlib cartopy python-louvain
   ```
3. **Add data** files:
   - Place `airports.dat` and `routes.dat` in the same folder where `flight_network.py` is located (or adjust file paths within the scripts).
4. **Run** the main script:
   ```bash
   python flight_network.py
   ```
   - This will generate:
     - Network statistics,
     - Centrality measures,
     - Community detection plots,
     - Robustness analysis charts,
     - Output visuals in the `output/` folder.

5. **Optional:** Run the SIR model separately:
   ```bash
   python disease_spread.py
   ```
   - Produces a basic SIR simulation plot saved as `SIR_Simulation_Result.png`.

### Key Features

- **Centrality Computations:** Quickly identify top hubs by degree, betweenness, or eigenvector metrics.
- **Community Detection:** Reveal how airports cluster regionally (Louvain algorithm).
- **Robustness Tests:** Remove top-degree hubs to see the network’s fragmentation and behavior under targeted attacks.
- **Disease Simulation:** Model how an epidemic might spread if certain airports become infected.

### Future Improvements

- **Integrate Actual Passenger Data:** Replace or supplement route counts with real passenger volumes or seat capacities.
- **Multi-Hub Attack Scenarios:** Expand beyond simple degree-based removal or incorporate betweenness-based strategies.
- **Advanced Epidemiological Models:** Switch from SIR to SEIR or multi-strain dynamics; incorporate flight schedules over time.
