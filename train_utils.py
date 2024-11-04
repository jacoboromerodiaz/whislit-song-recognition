from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

def parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Train whistle-song similarity model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file for this pipeline.",
    )

    return parser.parse_args()