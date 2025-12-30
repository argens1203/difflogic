from experiment.experiment import Experiment
from experiment.helpers.context import MultiContext

if __name__ == "__main__":
    m_ctx = MultiContext()

    m_ctx.add(Experiment.debug(dataset="iris", small=True))
    m_ctx.add(Experiment.debug(dataset="monk1", small=True))
    m_ctx.add(Experiment.debug(dataset="monk2", small=True))
    m_ctx.add(Experiment.debug(dataset="monk3", small=True))
    m_ctx.add(Experiment.debug(dataset="breast_cancer", small=True))
    m_ctx.add(Experiment.debug(dataset="compas", small=True))
    m_ctx.add(Experiment.debug(dataset="lending", small=True))

    m_ctx.display()
    m_ctx.to_csv(filename="results.csv", with_timestamp=True)
