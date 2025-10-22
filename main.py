from sqlite3 import Date
from experiment.experiment import Experiment
from experiment.helpers.context import MultiContext

if __name__ == "__main__":
    # experiment.run_with_cmd()
    # experiment.get_and_retest_model()

    # experiment.run_with_cmd()
    m_ctx = MultiContext()

    # m_ctx.add(Experiment.debug(dataset="iris", ohe=True))

    # m_ctx.add(Experiment.debug(dataset="iris", small=True))
    # m_ctx.add(Experiment.debug(dataset="monk1", small=True))
    # m_ctx.add(Experiment.debug(dataset="monk2", small=True))
    # m_ctx.add(Experiment.debug(dataset="monk3", small=True))
    # m_ctx.add(Experiment.debug(dataset="breast_cancer", small=True))
    # m_ctx.add(Experiment.debug(dataset="compas", small=True))
    # m_ctx.add(Experiment.debug(dataset="lending", small=True))

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

    # hitman_choices = ["sorted", "lbx", "sat"]
    # h_solver_choices = ["mgh", "cd195", "g3"]

    # for h_type in hitman_choices:
    #     if h_type == "sat":
    #         for h_solver in h_solver_choices:
    #             try:
    #                 m_ctx.add(
    #                     Experiment.debug(
    #                         dataset="lending", h_type=h_type, h_solver=h_solver
    #                     )
    #                 )
    #             except Exception as e:
    #                 print(f"Error with hitman={h_type}, h_solver={h_solver}: {e}")
    #     else:
    #         try:
    #             m_ctx.add(Experiment.debug(dataset="lending", h_type=h_type))
    #         except Exception as e:
    #             print(f"Error with hitman={h_type}: {e}")

    # m_ctx.add(Experiment.debug(dataset="monk2", explain_algorithm="both"))

    # for dataset in [
    #     "iris",
    #     "monk1",
    #     "monk2",
    #     "monk3",
    #     "breast_cancer",
    #     "compas",
    #     "lending",
    # ]:
    #     for exp_algorithm in ["var", "mus"]:
    #         for alpha in [1.5, 2.0, 2.5, 3.0]:
    #             for window in [10, 20, 30]:
    #                 # for exp_algorithm in ["mus", "mcs", "both", "find_one"]:
    #                 try:
    #                     m_ctx.add(
    #                         Experiment.debug(
    #                             dataset=dataset,
    #                             explain_algorithm=exp_algorithm,
    #                             alpha=alpha,
    #                             window=window,
    #                         )
    #                     )
    #                 except Exception as e:
    #                     print(
    #                         f"Error with dataset={dataset}, explain_algorithm={exp_algorithm}: {e}"
    #                     )

    # m_ctx.add(Experiment.debug(dataset="lending"))
    # for round in [0, 1, 2, 3]:
    # m_ctx.add(
    #     Experiment.debug(
    #         dataset="mnist",
    #         explain_algorithm="explain_one",
    #         custom=False,
    #         small=False,
    #         proc_rounds=1,
    #     )
    # )

    # m_ctx.add(
    #     Experiment.debug(
    #         dataset="lending",
    #         explain_algorithm="mus",
    #         solver_type="g3",
    #         h_solver="mgh",
    #         h_type="sat",
    #         custom=False,
    #         small=False,
    #     )
    # )

    for dataset in [
        "breast_cancer",
        "adult",
        "monk1",
        "monk2",
        "monk3",
        "iris",
        "compas",
        "lending",
    ]:
        for solver in [
            "cd",
            "cd15",
            "cd19",
            "cms",
            "gc3",
            "gc4",
            "g3",
            "g4",
            "g42",
            "lgl",
            "mcb",
            "mcm",
            "mpl",
            "mg3",
            "mc",
            "m22",
            "mgh",
        ]:
            try:
                for i in range(10):
                    m_ctx.add(
                        Experiment.debug(
                            dataset=dataset,
                            explain_algorithm="mus",
                            solver_type=solver,
                            h_solver="mgh",
                            h_type="sat",
                            custom=False,
                            small=True,
                        )
                    )
            except Exception as e:
                print(f"Error with solver={solver}: {e}")

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

    m_ctx.to_csv(filename=f"717bc2b.csv", with_timestamp=True)
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
