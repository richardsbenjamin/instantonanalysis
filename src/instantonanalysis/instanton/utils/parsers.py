from __future__ import annotations

from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

CALC_PARSER_ARGS = {
    "config-name": {
        "type": str,
        "required": True,
        "help": "Name of the config file to use."
    },
    "overrides": {
        "type": str,
        "nargs": "*",
        "required": False,
        "default": [],
        "help": "Hydra overrides, e.g. --overrides analysis.event_quantile=0.95"
    },
}

def get_arg_parser(description: str, args_dict: dict[str, dict]) -> Namespace:
    arg_parser = ArgumentParser(description=description)
    for arg, arg_params in args_dict.items():
        kwargs = {
            "type": arg_params["type"],
            "required": arg_params["required"],
            "help": arg_params["help"],
            "default": arg_params.get("default"),
        }
        if "nargs" in arg_params:
            kwargs["nargs"] = arg_params["nargs"]
        arg_parser.add_argument(f'--{arg}', **kwargs)
    run_args = arg_parser.parse_args()
    return run_args

def get_calc_args() -> Namespace:
    return get_arg_parser("Calculate instanton metrics.", CALC_PARSER_ARGS)
