from azureml.core import Workspace, Dataset, Experiment, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import InteractiveLoginAuthentication

# ----------- CONFIGURATION -----------
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "NLP8-2025"
compute_name = "adsai-lambda-0"

# ----------- AUTHENTICATION -----------
interactive_auth = InteractiveLoginAuthentication()
ws = Workspace(subscription_id, resource_group, workspace_name, auth=interactive_auth)


compute_target = ComputeTarget(workspace=ws, name="adsai-lambda-0")
env = Environment.get(workspace=ws, name="nlp8-env2", version="5")
run_config = RunConfiguration()
run_config.environment = env
 
train_data = Dataset.get_by_name(ws, name='dataset_train')
val_data = Dataset.get_by_name(ws, name='dataset_val')
 
train_step = PythonScriptStep(
    name="Train and Evaluate Model",
    script_name="train_script.py",
    arguments=[
        "--train_data", train_data.as_named_input("train_data").as_mount(),
        "--val_data", val_data.as_named_input("val_data").as_mount(),
        "--model_dir", "transformer-roberta-model",  # Pass ONLY the model name here!
        "--output_dir", "outputs"
    ],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False
)
 
pipeline = Pipeline(workspace=ws, steps=[train_step])
experiment = Experiment(workspace=ws, name="evaluate_model_experiment")
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)