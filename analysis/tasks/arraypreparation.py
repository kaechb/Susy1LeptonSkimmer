# coding: utf-8

import os
import law
import order as od
import luigi
import numpy as np
import operator
import sklearn.model_selection as skm

# other modules
from tasks.basetasks import ConfigTask
from tasks.coffea import CoffeaProcessor


class ArrayNormalisation(ConfigTask):

    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """

    channel = luigi.Parameter(default="N1b_CR", description="channel to prepare")

    def requires(self):
        return {
            "debug": CoffeaProcessor.req(self, processor="ArrayExporter", debug=True, workflow="local"),
            # comment out following line if just interested in debug
            "complete": CoffeaProcessor.req(self, processor="ArrayExporter"),
        }

    def path(self):
        return "/nfs/dust/cms/group/susy-desy/Susy1Lepton/0b/Run2_pp_13TeV_2016/CoffeaProcessor/testDNN/0b/ArrayExporter"

    def output(self):
        out = {}
        out.update(
            {  # "norm_values": self.local_target("norm_values.npy"),
                "oneHotLabels": self.local_target("oneHotLabels.npy"),
                "dataMerged": self.local_target("dataMerged.npy"),
                "meanStds": self.local_target("meanStds.npy"),
                "train": self.local_target("train.npy"),
                "trainLabels": self.local_target("trainLabels.npy"),
                "validation": self.local_target("validation.npy"),
                "validationLabels": self.local_target("validationLabels.npy"),
                "test": self.local_target("test.npy"),
                "testLabels": self.local_target("testLabels.npy"),
            }
        )
        return out

    def normalise(self, array):
        return ((array - array.mean()) / array.std(), array.mean(), array.std())

    def calc_norm_parameter(self, data):
        # return values to shift distribution to normal
        dat = np.swapaxes(data, 0, 1)
        means, stds = [], []
        for var in dat:
            means.append(var.mean())
            stds.append(var.std())
        return np.array(means), np.array(stds)

    def run(self):
        self.output()["oneHotLabels"].parent.touch()
        # load inputs from ArrayExporter
        targetDict = self.input()["complete"]["collection"].targets
        # make regular dict out of ordered dict
        fileDict = {}
        for i in targetDict.keys():
            fileDict.update(targetDict[i])
        procDict = {}
        oneHotLabels = []
        # loop through datasets and sort according to aux template
        datasets = list(fileDict.values())
        # os.listdir(self.path())
        for i, key in enumerate(self.config_inst.get_aux("DNN_process_template").keys()):
            processList = []
            for cat in self.config_inst.categories.names():
                for subproc in self.config_inst.get_aux("DNN_process_template")[key]:
                    # catch bottom level processes that have no childs
                    for dat in datasets:
                        if cat in dat.path and subproc in dat.path:
                            processList.append(dat.load())
                        for subsubproc in self.config_inst.get_process(subproc).walk_processes():
                            if cat in dat.path and subsubproc[0].name in dat.path:
                                processList.append(dat.load())
            procDict.update({key: np.concatenate(processList)})
            # 3 processes
            labels = np.zeros((len(np.concatenate(processList)), 3))
            labels[:, i] = 1
            oneHotLabels.append(labels)
        # merge all processes
        dataMerged = np.concatenate(list(procDict.values()))
        oneHotLabels = np.concatenate(oneHotLabels)
        # split up test set 9:1
        train, test, trainLabels, testLabels = skm.train_test_split(dataMerged, oneHotLabels, test_size=0.10, random_state=1)
        # train and validation set 80:20 FIXME
        train, validation, trainLabels, validationLabels = skm.train_test_split(train, trainLabels, test_size=0.5, random_state=2)
        # define means and stds for each variable
        means, stds = self.calc_norm_parameter(dataMerged)
        meanStds = np.vstack((means, stds))
        # save all arrays away, using the fact that keys have the variable name
        for key in self.output().keys():
            self.output()[key].dump(eval(key))


class CrossValidationPrep(ConfigTask):
    kfold = luigi.IntParameter(default=5)
    """
    Task to modify the Arrays produced by coffea
    Current idea: normalise them to prepare for the DNN
    """
    channel = luigi.Parameter(default="N1b_CR", description="channel to prepare")

    def requires(self):
        return CoffeaProcessor.req(self, processor="ArrayExporter")

    def output(self):
        out = {}
        out.update(
            {
                "cross_val_{}".format(i): {
                    "cross_val_train_{}".format(i): self.local_target("cross_val_train_{}.npy".format(i)),
                    "cross_val_trainLabels_{}".format(i): self.local_target("cross_val_trainLabels_{}.npy".format(i)),
                    "cross_val_validation_{}".format(i): self.local_target("cross_val_validation_{}.npy".format(i)),
                    "cross_val_validationLabels_{}".format(i): self.local_target("cross_val_validationLabels_{}.npy".format(i)),
                }
                for i in range(self.kfold)
            }
        )
        return out

    def normalise(self, array):
        return ((array - array.mean()) / array.std(), array.mean(), array.std())

    def calc_norm_parameter(self, data):
        # return values to shift distribution to normal
        dat = np.swapaxes(data, 0, 1)
        means, stds = [], []
        for var in dat:
            means.append(var.mean())
            stds.append(var.std())
        return np.array(means), np.array(stds)

    def run(self):
        # load inputs from ArrayExporter
        targetDict = self.input()["collection"].targets
        # make regular dict out of ordered dict
        fileDict = {}
        for i in targetDict.keys():
            fileDict.update(targetDict[i])
        procDict = {}
        oneHotLabels = []
        # loop through datasets and sort according to aux template
        datasets = list(fileDict.values())
        # at some point, we have to include k fold cross validation
        # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
        # from sklearn.model_selection import KFold
        for i, key in enumerate(self.config_inst.get_aux("DNN_process_template").keys()):
            processList = []
            for cat in self.config_inst.categories.names():
                for subproc in self.config_inst.get_aux("DNN_process_template")[key]:
                    # catch bottom level processes that have no childs
                    for dat in datasets:
                        if cat in dat.path and subproc in dat.path:
                            # self.config_inst.datasets.names():
                            processList.append(dat.load())
                        for subsubproc in self.config_inst.get_process(subproc).walk_processes():
                            if cat in dat.path and subsubproc[0].name in dat.path:
                                # load input
                                processList.append(dat.load())

            procDict.update({key: np.concatenate(processList)})
            labels = np.zeros((len(np.concatenate(processList)), 3))
            labels[:, i] = 1
            oneHotLabels.append(labels)
        # merge all processes
        dataMerged = np.concatenate(list(procDict.values()))
        oneHotLabels = np.concatenate(oneHotLabels)
        kfold = skm.KFold(n_splits=self.kfold, shuffle=True, random_state=42)
        # kfold returns generator, loop over generated indices
        # for each kfold, dump the respective data and labels
        for i, idx in enumerate(kfold.split(dataMerged)):
            # unpack tuple
            train_idx, val_idx = idx
            self.output()["cross_val_{}".format(i)]["cross_val_train_{}".format(i)].dump(dataMerged[train_idx])
            self.output()["cross_val_{}".format(i)]["cross_val_trainLabels_{}".format(i)].dump(oneHotLabels[train_idx])
            self.output()["cross_val_{}".format(i)]["cross_val_validation_{}".format(i)].dump(dataMerged[val_idx])
            self.output()["cross_val_{}".format(i)]["cross_val_validationLabels_{}".format(i)].dump(oneHotLabels[val_idx])
