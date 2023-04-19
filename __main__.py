# __main__.py

import sys
from .postpruner import run

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True, choices=[
        "mnli",
        "qqp",
        "qnli",
        "sst2",
        "stsb",
        "mrpc",
        "squad",
        "squad_v2",
    ])
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=0)

    parser.add_argument("--metric", type=str, choices=[
        "mac",
        "latency",
    ], default="mac")
    parser.add_argument("--constraint", type=float, required=True,
        help="MAC/latency constraint relative to the original model",
    )
    parser.add_argument("--mha_lut", type=str, default=None)
    parser.add_argument("--ffn_lut", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head_mask_output", type=str, default="head_mask.pt")
    parser.add_argument("--neuron_mask_output", type=str, default="neuron_mask.pt")
    args = parser.parse_args()

    run(
        model_name=args.model_name,
        task_name=args.task_name,
        ckpt_dir=args.ckpt_dir,
        constraint=args.constraint,
        output_dir=args.output_dir,
        gpu=args.gpu,
        metric=args.metric,
        mha_lut=args.mha_lut,
        ffn_lut=args.ffn_lut,
        num_samples=args.num_samples,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
