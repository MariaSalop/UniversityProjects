import os
from azureml.core import Workspace, ScriptRunConfig, Experiment, Environment, ComputeTarget

print("Azure ML submission script started")

# Configurations
ws = Workspace.from_config()
compute_target = ComputeTarget(workspace=ws, name="adsai-lambda-0")
env = Environment.get(workspace=ws, name="nlp8-env3", version="2")

experiment_name = "emotion-hpo-experiment"
script_path = "automated_hpo_emotion_classifier.py"
input_csv = "core_dataset.csv"

if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Could not find {input_csv}")

src = ScriptRunConfig(
    source_directory=".",
    script=script_path,
    arguments=[
        "--csv_path", input_csv,
        "--n_trials", "20",
        "--max_per_emotion", "10",
        "--seed", "42",
        "--study_path", "optuna_emotion_hpo_study.pkl"
    ],
    compute_target=compute_target,
    environment=env  # Pass the environment here
)

experiment = Experiment(ws, experiment_name)
print("Submitting run to AzureML...")
run = experiment.submit(src)
print(f"Run submitted! Monitor here: {run.get_portal_url()}")
print(f"RunId: {run.id}")

run.wait_for_completion(show_output=True)
