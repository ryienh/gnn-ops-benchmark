"""
Stuff here
"""
import torch.utils.benchmark as benchmark
import torch

from itertools import permutations


# def make_sparse(input, sparsity):

#     # seed
#     torch.manual_seed(100)

#     k_th_quant = torch.topk(input, int(sparsity * input.shape[-1]), largest=False)[0][
#         :, -1:
#     ]
#     mask = input <= k_th_quant
#     input[mask] = 0
#     return input


def make_sparse(input, sparsity):
    print(f"Input shape: {input.shape}")
    k = round(sparsity * input.shape[-1])
    k_th_quant = torch.topk(input, k, largest=False)[0][:, -1:]
    bool_tensor = input <= k_th_quant
    input[bool_tensor] = 0
    return input


# def make_sparse(input, sparsity):
#     binomial = torch.distributions.binomial.Binomial(probs=1 - sparsity)
#     return input * binomial.sample(input.size()) * (1.0 / (1 - sparsity))


class OpBenchmark:
    def __init__(self):
        pass

    def benchmark_and_compare(self, A, B, fn):
        """
        A and B are lists of different inputs to function fn. Later consider concating inputs and extacting to support
        more than 2 inputs
        """
        # Compare takes a list of measurements which we'll save in results.
        results = []

        inputs = permutations(A, B)
        for a, b in zip(inputs):
            # label and sub_label are the rows
            # description is the column
            label = "Batched dot"
            sub_label = f"[{a.shape}, {b.shape}]"
            x = torch.ones((b, n))
            for num_threads in [1, 4, 16, 32]:
                results.append(
                    benchmark.Timer(
                        stmt="batched_dot_mul_sum(x, x)",
                        setup="from __main__ import batched_dot_mul_sum",
                        globals={"x": x},
                        num_threads=num_threads,
                        label=label,
                        sub_label=sub_label,
                        description="mul/sum",
                    ).blocked_autorange(min_run_time=1)
                )
                results.append(
                    benchmark.Timer(
                        stmt="batched_dot_bmm(x, x)",
                        setup="from __main__ import batched_dot_bmm",
                        globals={"x": x},
                        num_threads=num_threads,
                        label=label,
                        sub_label=sub_label,
                        description="bmm",
                    ).blocked_autorange(min_run_time=1)
                )

        compare = benchmark.Compare(results)
        compare.print()


class TorchBenchmark(OpBenchmark):
    def __init__(self):
        super().__init__()


class TorchScatterBenchmark(OpBenchmark):
    def __init__(self):
        super().__init__()


class TorchSparseBenchmark(OpBenchmark):
    def __init__(self):
        super().__init__()


class TorchClusterBenchmark(OpBenchmark):
    def __init__(self):
        super().__init__()


class TorchSplineConvBenchmark(OpBenchmark):
    def __init__(self):
        super().__init__()
