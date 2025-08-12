from experiment.util import TrainingArgs


class TrainingContext:
    def __init__(
        self,
        train_loader,
        validation_loader,
        test_loader,
        loss_fn,
        optim,
        args: TrainingArgs,
    ):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optim = optim
        self.num_iterations = args.num_iterations
        self.eval_freq = args.eval_freq
        self.training_bit_count = args.training_bit_count
