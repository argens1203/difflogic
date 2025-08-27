from experiment.experiment import Experiment

if __name__ == "__main__":
    # experiment.run_with_cmd()
    # experiment.get_and_retest_model()

    # experiment.run_with_cmd()
    # Experiment.debug(dataset="iris")
    # Experiment.debug(dataset="monk1")
    # Experiment.debug(dataset="monk2")
    # Experiment.debug(dataset="monk3")
    # Experiment.debug(dataset="breast_cancer")
    Experiment.debug(dataset="adult")
    # experiment.debug(dataset="mnist")

    # experiment.experiment(datasets=["adult", "mnist"], base_experiment_id=10000)
    # experiment.find_model()

    # datasets = ["iris", "monk1", "monk2", "monk3", "adult", "breast_cancer"]
    # experiment_ids = list(range(1, 7))
    # experiment.experiment(datasets=datasets, experiment_ids=experiment_ids)
