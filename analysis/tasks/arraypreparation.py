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

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    channel = luigi.Parameter(default="N0b_CR", description="channel to prepare")

    def requires(self):
        return {
            "debug": CoffeaProcessor.req(
                self, processor="ArrayExporter", debug=True, workflow="local"
            ),
            # comment out following line if just interested in debug
            "complete": CoffeaProcessor.req(self, processor="ArrayExporter"),
        }

    def output(self):
        return self.local_target("hists")

    def output(self):
        out = {
            cat + "_" + proc: self.local_target("normed_" + cat + "_" + proc + ".npy")
            for proc in self.config_inst.processes.names()
            for cat in self.config_inst.categories.names()
            if self.channel in cat and not "data" in proc
        }
        out.update(
            {  # "norm_values": self.local_target("norm_values.npy"),
                "one_hot_labels": self.local_target("one_hot_labels.npy"),
            }
        )
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

        # load inputs from ArrayExporter
        target_dict = self.input()["complete"]["collection"].targets
        debug_dict = self.input()["debug"]["collection"].targets

        # make regular dict out of ordered dict
        file_dict = {}
        for i in target_dict.keys():
            file_dict.update(target_dict[i])

        one_hot_scheme = [
            name for name in self.config_inst.processes.names() if "data" not in name
        ]

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

        check_if = 0

        for key in proc_dict.keys():
            # for arr_list in proc_dict[key]:
            if not self.channel in key:
                continue

            array = np.vstack([array.load() for array in proc_dict[key]])

            print(key)
            process = key.replace(self.channel + "_", "")
            position = one_hot_scheme.index(process)
            labels = np.zeros((len(one_hot_scheme), len(array)))
            labels[position] = 1

            # this is sooo ugly, but unlike lists, appending numpy arrays requires a filling beforehand
            if check_if == 0:
                one_hot_labels = labels
                check_if += 1

            else:
                one_hot_labels = np.append(one_hot_labels, labels, axis=1)

            # change axis for normalisation, overwrite array to be memory friendly
            # save values for normalisation layer
            # I should not norm each process on their own I guess
            # norm_values = []
            # array = np.moveaxis(array, 1, 0)
            # for i, arr in enumerate(array):
            #    array[i], mean, std = self.normalise(arr)
            #    norm_values.append([mean, std])

            # roll axis back for dnn input
            # array = np.moveaxis(array, 1, 0)
            # np.save(self.output().parent.path + "/normed_" + key, array)

            self.output()[key].dump(array)

        # prepare one-hot encoded labels?
        categories = np.stack((np.ones(len(array)), np.zeros(len(array))))

        # self.output()["norm_values"].dump(norm_values)
        self.output()["one_hot_labels"].dump(one_hot_labels)

        # from IPython import embed
        # embed()
