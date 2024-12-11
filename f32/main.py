import os
import subprocess
from utils import get_argparser


def augment_argparser():
    parser = get_argparser()

    parser.add_argument(
        "--benchmark",
        type=str,
        help="The benchmark to run. Supports either full name (e.g. keyword_spotting) or abbreviation (e.g. kws)",
        choices=[
            "keyword_spotting",
            "image_classification",
            "visual_wake_words",
            "anomaly_detection",
            "kws",
            "ic",
            "vww",
            "ad",
        ],
        required=True,
    )

    return parser


def convert_args_to_string(args):
    parts = [
        f"--{arg}" if isinstance(value, bool) and value else f"--{arg} {value}"
        for arg, value in vars(args).items()
        if arg != "benchmark" and value is not False and value is not None
    ]

    return " ".join(parts)

benchmark_mappping = {
    "kws": "keyword_spotting",
    "ic": "image_classification",
    "vww": "visual_wake_words",
    "ad": "anomaly_detection",
}

def main():
    args = augment_argparser().parse_args()

    if args.benchmark in benchmark_mappping:
        args.benchmark = benchmark_mappping[args.benchmark]

    path = os.path.join(os.path.dirname(__file__), f"{args.benchmark}", "eval_fp32.py")

    cmd = f"python {path} {convert_args_to_string(args)}"
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
