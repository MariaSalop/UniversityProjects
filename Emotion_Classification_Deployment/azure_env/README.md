# Azure ML Environment Setup

This project sets up and registers a reproducible Conda environment in Azure Machine Learning based on a `conda.yaml` file.

---

## Files Included

- `conda.yaml` — Defines the Conda environment with all necessary dependencies.  
- `register_env.py` — Python script to register the environment in Azure ML.  
- `config.json` — Workspace configuration file (used if not authenticating via Azure CLI).  

---

## Requirements

- Azure ML Workspace  
- Conda installed (Miniconda or Anaconda)  
- Python 3.9  
- Azure CLI (for authentication via `az login`)  
- `azureml-core` installed in the active environment:

```bash
pip install azureml-core
```

---

## Step 1 — Create the Conda Environment

Make sure you are in the same directory as `conda.yaml`, then run:

```bash
conda env create -f conda.yaml
```

Activate the environment:

```bash
conda activate nlp8-env
```

---

## Step 2 — Register the Environment in Azure ML (using `config.json`)

1. Ensure the following files are in the same folder:  
   - `conda.yaml`  
   - `register_env.py`  
   - `config.json`  

2. Fill in `config.json` with your Azure ML details:

```json
{
  "subscription_id": "your-subscription-id",
  "resource_group": "your-resource-group",
  "workspace_name": "your-workspace-name"
}
```

3. Run the registration script:

```bash
python register_env.py
```

If successful, you will see:

```
Environment registered successfully in Azure ML.
```
