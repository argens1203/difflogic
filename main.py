from lgn.experiment import Experiment

if __name__ == "__main__":
    experiment = Experiment()
    # experiment.get_and_retest_model()

    # experiment.run_with_cmd()
    experiment.debug(dataset="adult")
    experiment.debug(dataset="breast_cancer")
    experiment.debug(dataset="mnist")
    # experiment.find_model()

    # datasets = ["iris", "monk1", "monk2", "monk3", "adult", "breast_cancer"]
    # experiment_ids = list(range(1, 7))
    # experiment.experiment(datasets=datasets, experiment_ids=experiment_ids)
