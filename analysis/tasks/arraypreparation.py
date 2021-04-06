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

    def output(self):
        out = {
            cat + "_" + proc: self.local_target("normed_" + cat + "_" + proc + ".npy")
            for proc in self.config_inst.processes.names()
            for cat in self.config_inst.categories.names()
        }
        return out

    def normalise(self, array):
        return ((array - array.mean()) / array.std(), array.mean(), array.std())

    def run(self):

        # self.output().parent.touch()
        # inp = self.input()["collection"][0].load()
        # array = self.input()["collection"].targets[0]["N0b_TTZ_qq"].load()
        # path=self.input()["collection"].targets[0].path

        # print(self.config_inst.variables.names(), ":")
        # print(array)

        target_dict = self.input()["complete"]["collection"].targets

        # make regular dict out of ordered dict
        file_dict = {}
        for i in target_dict.keys():
            file_dict.update(target_dict[i])

        proc_dict = {}
        for cat in self.config_inst.categories.names():
            for proc in self.config_inst.processes:
                if "data" in proc.name:
                    continue
                target_list = []
                for child in proc.walk_processes():
                    key = cat + "_" + child[0].name
                    if key in file_dict.keys():

                        target_list.append(file_dict[key])
                proc_dict.update({cat + "_" + proc.name: target_list})

        for key in proc_dict.keys():
            # for arr_list in proc_dict[key]:
            array = np.vstack([array.load() for array in proc_dict[key]])

            # change axis for normalisation, overwrite array to be memory friendly
            # save values for normalisation layer
            norm_values = []
            array = np.moveaxis(array, 1, 0)
            for i, arr in enumerate(array):
                array[i], mean, std = self.normalise(arr)
                norm_values.append([mean, std])

            # roll axis back for dnn input
            array = np.moveaxis(array, 1, 0)
            # np.save(self.output().parent.path + "/normed_" + key, array)

            self.output()[key].dump(array)

        # prepare one-hot encoded labels?
        categories = np.stack((np.ones(len(array)), np.zeros(len(array))))

        # from IPython import embed

        # embed()
