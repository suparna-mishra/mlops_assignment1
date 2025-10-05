# ML Ops Assignment 1 — House Price Prediction
This repository contains the solution for **ML Ops Assignment 1**. 
We build and automate a machine learning workflow to predict house prices using the **Boston Housing** dataset.  
Two classical ML models are implemented:

- `DecisionTreeRegressor` (in `train.py`)
- `KernelRidge` (in `train2.py`)
-  A shared `misc.py` file contains generic functions for loading data, training, and evaluation.

#Branches:
- `main` — contains this README and merged code.
- `dtree` — Decision Tree model code (merged into main).
- `kernelridge` — Kernel Ridge model code and GitHub Actions workflow.

#This README covers:  
- How to install dependencies  
- How to run both scripts  
- CI/CD workflow info  


#Setup
```bash
conda create -n mlops python=3.10 -y
conda activate mlops
pip install -r requirements.txt
```

#Running Locally
```bash
python train.py
python train2.py
```

Both scripts will print the Mean Squared Error (MSE) on the test set

#GitHub Actions (CI/CD)

A GitHub Actions workflow (.github/workflows/mlops.yml) is set up to automatically:

-Trigger on push to the kernelridge branch.

-Checkout the code.

-Install dependencies.

-Run train.py and train2.py.

-Display MSE results in the Actions logs.


To see it : Push changes to the kernelridge branch



