# Dempster-Shafer Evidential Spiking Neural Network Framework (DS-SNN)
- Currently has not cross-dataset validation other than OASIS-2
- Based entirely on clinical + MRI BBA assignments on OASIS-2
- MRI Pre-processing scripts are in place, but has not yet pre-processed the RAW files
- Proof-of-Concept study for the larger DS-SNN for multi-modal fusion (DS-SNN-MF)

## What does each module do
Most of the module names are fairly self-explanatory.
- `DSModule` handles Dempster-Shafer Theory related computations
- `SNNModule` handles the MRI processing through the LIF-SNN layer
- `TrainerModule` handles the training process
- `OasisModule` handles loading and normalizing the OASIS-2 data
- `config` maintains a single configuration class object to be used by all the modules
- `main` is the primary orechestrator module that runs the entire pipeline
- Other files include:
    - `mri_preprocess` is the the MRI pre-processing layer that has not been tried yet
    - `leaky_demo` produces a standard MLP baseline on a leaky run of the OASIS-2 clinical data
    - `baselines` produces a primitive Machine Learning (LR, RF) baseline on the dataset
    - `sanity_overfit` is a leftover testing script that was made to check if everything was working

## How to use
For most purposes, run `python3 main.py` from the root directory.
`.\docs\` include documents with further details regarding the overall mechanism.




---

This repository is maintained by Abrar Fahyaz (University of Dhaka).

This is a work in progress. The main idea and formulations used in this project are from Dr. Maybin K. Muyeba (University of Salford).