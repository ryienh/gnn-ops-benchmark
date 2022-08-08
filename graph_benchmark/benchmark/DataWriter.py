import pandas as pd
import os


class DataWriter:
    def __init__(self, op_name, param_names=None):
        self.__data = []
        self.__name = op_name
        self.__param_names = param_names

    def add_param_names(self, param_names):
        self.__param_names = param_names

    # Takes raw strings, formats, appends to __data
    def add_entry(self, params_lst, tshape, sparsity, bm_val, delimiter=";"):
        params = delimiter.join(params_lst)
        input_dims = str(tshape)
        self.__data.append([params, input_dims, sparsity, bm_val])

    def write_data(
        self,
        path=None,
    ):

        columns = (
            [
                self.__param_names,
                "Input size (>95% mem util)*",
                "Sparsity",
                "GPU clock time",
            ],
        )
        df = pd.DataFrame(self.__data, columns=columns[0])

        if path is not None:
            df.to_csv(os.path.join(path, f"{self.__name}.csv"))
