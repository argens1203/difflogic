from sqlite3 import Date
from experiment.experiment import Experiment
from experiment.helpers.context import MultiContext

if __name__ == "__main__":
    # experiment.run_with_cmd()
    # experiment.get_and_retest_model()

    # experiment.run_with_cmd()
    m_ctx = MultiContext()

    # m_ctx.add(Experiment.debug(dataset="iris", ohe=True))

    m_ctx.add(Experiment.debug(dataset="iris", small=True))
    m_ctx.add(Experiment.debug(dataset="monk1", small=True))
    m_ctx.add(Experiment.debug(dataset="monk2", small=True))
    m_ctx.add(Experiment.debug(dataset="monk3", small=True))
    m_ctx.add(Experiment.debug(dataset="breast_cancer", small=True))
    m_ctx.add(Experiment.debug(dataset="compas", small=True))
    m_ctx.add(Experiment.debug(dataset="lending", small=True))

    # m_ctx.add(Experiment.debug(dataset="mnist", small=True))

    # # m_ctx.add(Experiment.debug(dataset="adult", small=True))
    # # m_ctx.add(Experiment.debug(dataset="adult", parent=True, small=True))

    # m_ctx.add(Experiment.debug(dataset="adult"))

    # m_ctx.add(Experiment.debug(dataset="compas"))

    # choices = [
    #     # "pw",
    #     # "seqc",
    #     # "cardn",
    #     # "sortn",
    #     # "tot",
    #     "mtot",
    #     # "kmtot",
    #     # "bit",
    #     # "lad",
    #     # "native",
    # ]
    # # choices = ["pw", "seqc", "cardn", "sortn", "tot", "mtot", "kmtot"]
    # for cho1 in choices:
    #     for cho2 in ["pw"]:
    #         try:
    #             m_ctx.add(
    #                 Experiment.debug(
    #                     dataset="lending", enc_type_at_least=cho1, enc_type_eq=cho2
    #                 )
    #             )
    #         except Exception as e:
    #             print(f"Error with enc_type_at_least={cho1}, enc_type_eq={cho2}: {e}")

    # m_ctx.add(Experiment.debug(dataset="lending"))
    # m_ctx.add(Experiment.debug(dataset="mnist"))

    # # m_ctx.add(Experiment.debug(dataset="adult"))
    # # m_ctx.add(Experiment.debug(dataset="breast_cancer", ohe=True))
    # # m_ctx.add(Experiment.debug(dataset="adult", ohe=True))

    # m_ctx.add(Experiment.debug(dataset="iris", reverse=True))
    # m_ctx.add(Experiment.debug(dataset="monk1", reverse=True))
    # m_ctx.add(Experiment.debug(dataset="monk2", reverse=True))
    # m_ctx.add(Experiment.debug(dataset="monk3", reverse=True))
    # m_ctx.add(Experiment.debug(dataset="breast_cancer", reverse=True))

    # # m_ctx.add(Experiment.debug(dataset="adult", reverse=True))
    # # m_ctx.add(Experiment.debug(dataset="mnist"))

    m_ctx.display()

    m_ctx.to_csv(filename=f"32d203a.csv", with_timestamp=True)
    # # Experiment.debug(dataset="monk1")
    # # Experiment.debug(dataset="monk2")
    # # Experiment.debug(dataset="monk3")
    # # Experiment.debug(dataset="breast_cancer")
    # # Experiment.debug(dataset="adult")
    # # Experiment.debug(dataset="mnist")

    # # experiment.experiment(datasets=["adult", "mnist"], base_experiment_id=10000)
    # # experiment.find_model()

    # # datasets = ["iris", "monk1", "monk2", "monk3", "adult", "breast_cancer"]
    # # experiment_ids = list(range(1, 7))
    # # experiment.experiment(datasets=datasets, experiment_ids=experiment_ids)
