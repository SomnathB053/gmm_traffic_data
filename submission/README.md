# Clustering Large Datasets with Gaussian Mixture Models for Traffic Analysis

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

## How to run

* Setup the environment
    - Ensure python is installed in the system. This project was run in Python==3.11.
    - In the root folder, create a virtual environment. Open the command-line/terminal in root and run `python -m venv .venv` or `python3 -m venv .venv`.
    - Source the virtual environment using 
        - `. .venv/bin/activate` on Linux
        -  `.\.venv\Scripts\Activate.PS1` on Windows PowerShell.
    - Install required packages as `pip install -r requirements.txt`
* Files
    We have implemented two separate approaches to cluster the datasets, same provided as notebooks and scripts.
    - `sensors_clusters.ipynb` (Notebook) or `sensors_clusters.py` (Script) on Dataset `PEMSD08`
    - `single_sensor_traffic_cluster.ipynb` (Notebook) or `single_sensor_traffic_cluster.py` (Script) on Dataset `PEMSD04`

* To run the python scripts:
    - Setup the environment as previously stated.
    - Run `python <file_name>.py` on the terminal/command_prompt/powershell individually.

The output is displayed on the terminal and as matplotlib figures.


