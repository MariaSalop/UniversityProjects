import yaml
from pathlib import Path

PIPELINE_YAML = Path("azureml/pipeline_retrain.yml")


def test_pipeline_structure():
    data = yaml.safe_load(PIPELINE_YAML.read_text())
    jobs = data["jobs"]
    # check that all four steps are present
    assert set(jobs) == {"prep", "train", "eval", "register"}

    # make sure that the train → eval outputs link is prescribed
    assert "${{parent.jobs.train.outputs.metrics_out}}" in str(jobs["eval"])
