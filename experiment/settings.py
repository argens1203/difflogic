from experiment.args.model_args import ModelArgs


class Settings:
    # Model training hyper parameters (according to the paper)
    all_temperatures = [1, 1 / 0.3, 1 / 0.1, 1 / 0.03, 1 / 0.01]
    learning_rate = 0.01
    epoch = 200
    batch_size = 100

    # Dataset specific hyper parameters (according to the paper)
    network_param = {
        "adult": {
            "num_neurons": 256,
            "num_layers": 5,
            "num_iterations": 200,
            "batch_size": 100,
            #        "temperature": None,
        },
        "monk1": {
            "num_neurons": 24,
            "num_layers": 6,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "monk2": {
            "num_neurons": 12,
            "num_layers": 6,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "monk3": {
            "num_neurons": 12,
            "num_layers": 6,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "breast_cancer": {
            "num_neurons": 128,
            "num_layers": 5,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "mnist": {
            "small": {
                "num_neurons": 8000,
                "num_layers": 6,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "normal": {
                "num_neurons": 64000,
                "num_layers": 6,
                "num_iterations": 200,
                "batch_size": 100,
            },
        },
        "cifar10": {
            "small": {
                "num_neurons": 12000,
                "num_layers": 4,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "medium": {
                "num_neurons": 128000,
                "num_layers": 4,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "large": {
                "num_neurons": 256000,
                "num_layers": 5,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "largeX2": {
                "num_neurons": 512000,
                "num_layers": 5,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "largeX4": {
                "num_neurons": 1024000,
                "num_layers": 5,
                "num_iterations": 200,
                "batch_size": 100,
            },
        },
    }

    # Our setup for debugging
    dataset_neurons = [
        ("iris", 12),
        ("caltech101", 41 * 101),
        ("adult", 58),
        ("monk1", 10),
        ("monk2", 10),
        ("monk3", 10),
        ("breast_cancer", 26),
        ("mnist", 400),
        ("lending", 100),  # 182 in 2 out
        ("compas", 21),  # 34 in 3 out
    ]
    dataset_params = [
        (
            ds,
            {
                **{"num_neurons": neurons},
                **{
                    "num_layers": 2,
                    "num_iterations": 2000,
                    "batch_size": 100,
                },
            },
        )
        for ds, neurons in dataset_neurons
    ]
    debug_network_param = dict(dataset_params)

    @staticmethod
    def get_settings(
        dataset_name: str = "", paper=False, minimal=True
    ) -> dict[str, int]:
        if not paper:
            return Settings.debug_network_param.get(dataset_name, {})
        if dataset_name == "iris":
            return {
                "num_neurons": 24,
                "num_layers": 5,
            }  # Custom created large iris model
        if minimal:
            if dataset_name not in ["mnist", "cifar10"]:
                return Settings.network_param.get(dataset_name, {})
            else:
                return (Settings.network_param.get(dataset_name) or {}).get("small", {})
        return {}

    @staticmethod
    def get_model_args(dataset_name: str = "", paper=False) -> ModelArgs:
        params = Settings.get_settings(dataset_name, paper)
        return ModelArgs(
            num_neurons=params.get("num_neurons"),
            num_layers=params.get("num_layers"),
        )

    @staticmethod
    def get_model_path(dataset: str, size: str) -> str:
        if size == "small":
            return "model-paths/$" + dataset + "_" + "model.pth"
        if size == "debug":
            return dataset + "_" + "model.pth"

        assert False
