import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id

def _already_ran(entry_point_name, parameters, experiment_id=None):
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    print("-----------------Checking Run Info!----------------")
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        print("Getting Tags")
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            print("Getting Run value")
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                print("Got Run value but Match Failed")
                break
        if match_failed:
            print("No Run value, Match Failed")
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping (run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        return client.get_run(run_info.run_id)
    eprint("-----------------No matching run has been found----------------")
    return None

# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return MlflowClient().get_run(submitted_run.run_id)

def workflow():
    with mlflow.start_run() as active_run:
        print("Entering Train Run")
        train_run = _get_or_run("train", {"train_data_path": "..//train.csv"})
        print("Train run = ", train_run)
        print("Entering Validate Run")
        validate_run = _get_or_run(
            "validate", {"validate_data_path": "..//train.csv"}
        )
        print("Validate run = ", validate_run)
        mlflow.log_metric("main-metric", 1000.00)

if __name__ == "__main__":
    print("-----------------Entering workflow!----------------")
    workflow()
