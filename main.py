from experiment import Experiment

if __name__ == "__main__":
    experiment = Experiment()
    # experiment.run_with_cmd()

    datasets = ["iris", "monk1", "monk2", "monk3", "adult", "breast_cancer"]
    experiment_ids = list(range(1, 7))
    experiment.experiment(datasets=datasets, experiment_ids=experiment_ids)
