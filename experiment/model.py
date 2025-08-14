import time
import torch
import logging

from constant import device

from lgn.model import get_model, compile_model, train_eval, multi_eval

logger = logging.getLogger(__name__)


class Model:
    @staticmethod
    def train_model(args, model, loss_fn, optim, ctx):
        train_eval(
            args,
            ctx.train_loader,
            None,
            ctx.test_loader,
            model,
            loss_fn,
            optim,
            ctx.results,
        )

    @staticmethod
    def eval_model(args, model, ctx):
        multi_eval(
            model,
            ctx.train_loader,
            ctx.test_loader,
            None,
            results=ctx.results,
            packbits_eval=args.packbits_eval,
        )

    @staticmethod
    def get_model(args, ctx):
        model, loss_fn, optim = get_model(args, ctx.results)
        if args.save_model and args.load_model:
            try:
                model.load_state_dict(
                    torch.load(args.model_path, map_location=torch.device(device))
                )
                Model.eval_model(args, model, ctx)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.info(f"Error loading model: {e}; using saved model")
                Model.train_model(args, model, loss_fn, optim, ctx)
                Model.eval_model(args, model, ctx)
                torch.save(model.state_dict(), args.model_path)

        elif args.save_model:
            Model.train_model(args, model, loss_fn, optim, ctx)
            Model.eval_model(args, model, ctx)
            torch.save(model.state_dict(), args.model_path)

        elif args.load_model:
            model.load_state_dict(
                torch.load(args.model_path, map_location=torch.device(device))
            )
            Model.eval_model(args, model, ctx)

        else:
            Model.train_model(args, model, loss_fn, optim, ctx)
            Model.eval_model(args, model, ctx)

        ####################################################################################################################
        if ctx.results is not None:
            ctx.results.store_custom("model_complete_time", time.time())
            ctx.results.store_model_ready_time()

        if args.compile_model:
            compile_model(args, model, ctx.test_loader)

        ctx.results.store_model_ready_time()
        return model
