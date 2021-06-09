import logging
import argparse
from pprint import pprint
from kb_ae_bert.trainer.train import train
from kb_ae_bert.utils.config import *

logging.root.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    p_train = subparsers.add_parser("train", help="Start training.")

    p_train.add_argument(
        "--config", type=str, required=True, help="Path of the config file.",
    )

    p_generate = subparsers.add_parser(
        "generate", help="Generate an example configuration."
    )

    p_generate.add_argument(
        "--stages",
        type=str,
        required=True,
        help="Stages to execute. Example: qa,qa,kb_encode",
    )
    p_generate.add_argument(
        "--print", action="store_true", help="Direct config output to screen."
    )
    p_generate.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="JSON config file output path.",
    )

    args = parser.parse_args()
    if args.command == "train":
        train(load_config(args.config))

    elif args.command == "generate":
        config = Config()
        for stage in args.stages.split(","):
            if stage == "qa":
                config.pipeline.append(stage)
                config.configs.append(QATrainConfig())
            elif stage == "kb_encoder":
                config.pipeline.append(stage)
                config.configs.append(KBEncoderTrainConfig())
            else:
                raise ValueError(f"Unknown stage {stage}")

        if args.print:
            pprint(config.dict())
        else:
            save_config(config, args.output)
            print(f"Config saved to {args.output}")
