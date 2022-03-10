from graph_benchmark.profile.OpProfiler import OpProfiler
import click


@click.command()
@click.option(
    "--config",
    default="./prof_config.json",
    prompt="Enter relative path to config file.",
    help="Relative path to config file. Must be in json format.",
)
def main(config):

    # init Profiler
    my_profiler = OpProfiler(config)
    my_profiler.profile_models()


if __name__ == "__main__":
    main()
