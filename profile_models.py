from graph_benchmark.profile.OpProfiler import OpProfiler
import click

# import wandb


@click.command()
@click.option(
    "--config",
    default="./prof_config.json",
    prompt="Enter relative path to config file.",
    help="Relative path to config file. Must be in json format.",
)
def main(config):

    # wandb.init(project="gnn-kernel-benchmark")

    # init Profiler
    my_profiler = OpProfiler(config)
    my_profiler.profile_models()
    print("hello world!")


if __name__ == "__main__":
    main()
