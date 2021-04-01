# coding: utf-8

import os
import law
import order as od
import luigi
import coffea
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
from tqdm.auto import tqdm
import operator

# other modules
from tasks.basetasks import *
from tasks.coffea import CoffeaProcessor


class ArrayNormalisation(ConfigTask):
    def requires(self):
        return {
            "debug": CoffeaProcessor.req(
                self, processor="ArrayExporter", debug=True, workflow="local"
            ),
            "complete": CoffeaProcessor.req(self, processor="ArrayExporter"),
        }

    def output(self):
        return self.local_target("hists")

    def normalise(self, array):
        return ((array - array.mean()) / array.std(), array.mean(), array.std())

    def run(self):

        from IPython import embed

        embed()
        # inp = self.input()["collection"][0].load()
        array = self.input()["collection"].targets[0]["N0b_TTZ_qq"].load()
        # path=self.input()["collection"].targets[0].path

        print(self.config_inst.variables.names(), ":")
        print(array)

        # change axis for normalisation, overwrite array to be memory friendly
        # save values for normalisation layer
        norm_values = []
        array = np.moveaxis(array, 1, 0)
        for i, arr in enumerate(array):
            array[i], mean, std = self.normalise(arr)
            norm_values.append([mean, std])

        # roll axis back for dnn input
        array = np.moveaxis(array, 1, 0)

        categories = np.stack((np.ones(len(array)), np.zeros(len(array))))

        from IPython import embed

        embed()
