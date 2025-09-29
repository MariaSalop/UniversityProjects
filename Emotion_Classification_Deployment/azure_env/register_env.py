from azureml.core import Workspace, Environment

# Connect to your Azure ML workspace using config.json
ws = Workspace.from_config(path="config.json")

# Load your Conda environment definition from YAML
env = Environment.from_conda_specification(
    name="nlp8-env3",
    file_path="conda.yaml"
)

# Register the environment in Azure ML
env.register(workspace=ws)

print("Environment registered successfully in Azure ML.")

