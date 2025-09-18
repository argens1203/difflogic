from experiment.experiment import Experiment
from experiment.helpers.context import MultiContext

if __name__ == "__main__":
    # experiment.run_with_cmd()
    # experiment.get_and_retest_model()

    # experiment.run_with_cmd()
    m_ctx = MultiContext()
    m_ctx.add(Experiment.debug(dataset="iris"))
    m_ctx.add(Experiment.debug(dataset="monk1"))
    m_ctx.add(Experiment.debug(dataset="monk2"))
    m_ctx.add(Experiment.debug(dataset="monk3"))
    m_ctx.add(Experiment.debug(dataset="breast_cancer"))
    m_ctx.add(Experiment.debug(dataset="adult"))
    # m_ctx.add(Experiment.debug(dataset="mnist"))
    m_ctx.to_csv(filename="results_multi.csv")
    # Experiment.debug(dataset="monk1")
    # Experiment.debug(dataset="monk2")
    # Experiment.debug(dataset="monk3")
    # Experiment.debug(dataset="breast_cancer")
    # Experiment.debug(dataset="adult")
    # Experiment.debug(dataset="mnist")

    # experiment.experiment(datasets=["adult", "mnist"], base_experiment_id=10000)
    # experiment.find_model()

    # datasets = ["iris", "monk1", "monk2", "monk3", "adult", "breast_cancer"]
    # experiment_ids = list(range(1, 7))
    # experiment.experiment(datasets=datasets, experiment_ids=experiment_ids)
