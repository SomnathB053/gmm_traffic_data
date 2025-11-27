# Clustering Large Datasets with Gaussian Mixture Models for Traffic Analysis
# Submitted by Group 24 for CSL7620 project, IIT Jodhpur

## File structure
root/
├── sensor_clusters.py
├── single_sensor_traffic_cluster.py
├── requirements.txt
├── sensors_clusters.ipynb
├── single_sensor_traffic_cluster.ipynb
├── README.md
├── report.pdf
└── data/
    ├── PEMS08.npz
    └── PEMS04.npz

## File description
- PEMS08.npz: Dataset used in `sensor_clusters` approach.
- PEMS04.npz: Dataset used in `single_sensor_traffic_cluster` approach.
- requirements.txt: Lists all python requirements to be installed to run the programs.
- sensor_clusters.py: Python script file to run the entire Approach 1 pipeline end to end.
- sensors_clusters.ipynb: Jupyter notebook version for `sensor_clusters` for easy viewing.
- single_sensor_traffic_cluster.py: Python script file to run the entire Approach 2 pipeline end to end.
- single_sensor_traffic_cluster.ipynb: Jupyter notebook version for `single_sensor_traffic_cluster` for easy viewing.
- report.pdf: Final report summarizing project work.

Disclaimer: In the case the datasets are missing they can be downloaded from the following links and placed in the root project folder:
- [PEMS08](https://zenodo.org/records/7816008/files/PEMS08.npz?download=1)
- [PEMS04](https://zenodo.org/records/7816008/files/PEMS04.npz?download=1)

## How to run

* Setup the environment
    - Ensure python is installed in the system. This project was run in Python==3.11.
    - In the root folder, create a virtual environment. Open the command-line/terminal in root and run `python -m venv .venv` or `python3 -m venv .venv`.
    - Source the virtual environment using 
        - `. .venv/bin/activate` on Linux
        -  `.\.venv\Scripts\activate` on Windows PowerShell.
        If you encounter an error regarding permissions. Try to run `.\.Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
    - Install required packages as `pip install -r requirements.txt`
* Files
    We have implemented two approaches, spatial and temporal, to cluster the datasets, each provided as notebook and script format separately.
    - `sensors_clusters.ipynb` (Notebook) or `sensors_clusters.py` (Script) on Dataset `PEMSD08`
    - `single_sensor_traffic_cluster.ipynb` (Notebook) or `single_sensor_traffic_cluster.py` (Script) on Dataset `PEMSD04`
* To run the python scripts (to parse data, run model and display output):
    - Setup the environment as previously stated.
    - Run `python <file_name>.py` on the terminal/command_prompt/powershell individually.

(To run the notebooks you need to install jupyter notebook from [here](https://jupyter.org/install))

The output is displayed on the terminal and as matplotlib figures.


