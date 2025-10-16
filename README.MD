# VARADE++: an edge-friendly framework for real-time Anomaly Detection in production plants

This repository contains the necessary files and code to reproduce the experiments from the paper **"VARADE++: An Edge-Friendly Framework for Real-Time Anomaly Detection in Production Plants"**. The experiments utilize the TimeSeAD framework.

## Repository Structure

- **`timesead/`**: This folder contains the code of the VARADE++ and Naive methods, as well as the Kuka dataset in a format compatible with the TimeSeAD framework.
- **`timesead_experiments/`**: This folder contains the experiment configurations

## Usage

To use this repository, follow the steps below:

1. **Clone the TimeSeAD Framework**:
   ```bash
   git clone https://github.com/wagner-d/TimeSeAD.git

2. **Copy the Content of This Repository**:
   Copy the entire contents of this repository, including the folder structure, into the local clone of the TimeSeAD repository. This will integrate VARADE++ and other related experiments into TimeSeAD.

3. **Run the runKuka.py file**:
   The script will perform the training and evaluation loops for all the methods tested on the Kuka dataset, and return the AUROC, AUPRC and best F1 score of each method in the log folder.

4. **Refer to TimeSeAD Documentation**:
   For detailed instructions on using TimeSeAD, please refer to the [TimeSeAD documentation](https://timesead.readthedocs.io/en/latest/).
