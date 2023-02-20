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
import pickle
import sklearn as sk
import torch

# captum
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance


# other modules
from tasks.basetasks import ConfigTask, DNNTask, HTCondorWorkflow
from tasks.coffea import CoffeaProcessor, CoffeaTask
from tasks.arraypreparation import ArrayNormalisation
from tasks.pytorch_test import PytorchMulticlass

import utils.PytorchHelp as util


class PlotCoffeaHists(ConfigTask):

    """
    Plotting all Histograms produced by coffea
    Utility for doing log scale, data comparison and only debug plotting
    """

    logScale = luigi.BoolParameter()
    unblinded = luigi.BoolParameter()
    signalScale = luigi.IntParameter(default=1)
    debug = luigi.BoolParameter(default=False)
    debugDataset = luigi.Parameter(default="data_e_C")  # take a small set to reduce computing time
    debug_str = luigi.Parameter(default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root")  # FIXME Paths

    def requires(self):
        if self.debug:
            return CoffeaProcessor.req(self, processor="Histogramer", workflow="local", debug=True, debugDataset=self.debugDataset, debug_str=self.debug_str)
        else:
            return CoffeaProcessor.req(self, processor="Histogramer")

    def output(self):
        path = ""
        if self.logScale:
            path += "_log"
        if self.unblinded:
            path += "_data"
        if self.debug:
            path += "_debug"
        return self.local_target("hists{}.pdf".format(path))

    def run(self):
        inp = self.input()["collection"][0].load()
        self.output().parent.touch()
        uncStyle = {"label": "Stat. Unc.", "hatch": "///", "facecolor": "none", "edgecolor": (0, 0, 0, 0.5), "linewidth": 0}
        dataUncStyle = {"linestyle": "none", "marker": ".", "markersize": 10.0, "color": "k", "elinewidth": 1}
        line_opts = {
            "linestyle": "-",
            "color": "r",
        }
        # create pdf object to save figures on separate pages
        with PdfPages(self.output().path) as pdf:
            # plot each hist
            for var in tqdm(self.config_inst.variables, unit="variable"):
                hists = inp[var.name]
                categories = [h.name for h in hists.identifiers("category")]
                for cat in categories:
                    # lists for collecting and sorting hists
                    bgHists = []
                    histAttributes = []
                    dataHists = []
                    if self.unblinded:
                        # build canvas for data plotting
                        fig, (ax, rax) = plt.subplots(
                            2,
                            1,
                            figsize=(18, 10),
                            sharex=True,
                            gridspec_kw=dict(
                                height_ratios=(3, 1),
                                hspace=0,
                            ),
                        )
                    else:
                        fig, ax = plt.subplots(figsize=(18, 10))
                    for proc in self.config_inst.processes:
                        # unpack the different coffea hists and regroup them
                        if "data" in proc.name and self.unblinded:
                            childHists = hists[[p[0].name for p in proc.walk_processes()], cat]
                            mapping = {proc.label: [p[0].name for p in proc.walk_processes() if not "B" in p[0].name]}
                            grouped = childHists.group("dataset", coffea.hist.Cat("process", proc.label_short), mapping).integrate("category")
                            dataHists.append(grouped)
                        if not self.debug and not "data" in proc.name:
                            # for each process, map childs together to plot in one, get rid of cat axis
                            childHists = hists[[p[0].name for p in proc.walk_processes()], cat]
                            mapping = {proc.label: [p[0].name for p in proc.walk_processes()]}
                            grouped = childHists.group("dataset", coffea.hist.Cat("process", proc.label_short), mapping).integrate("category")
                            bgHists.append(grouped)
                            histAttributes.append(proc.color)
                        if self.debug:
                            dat = hists.identifiers("dataset")
                            bg = hists[(str(dat[0]), cat)].integrate("category")
                    if not self.debug:
                        bg = bgHists[0]
                        for i in range(1, len(bgHists)):
                            bg.add(bgHists[i])

                    # order the processes by magnitude of integral
                    order = bg.integrate(var.name).values()
                    bgOrder = sorted(order.items(), key=operator.itemgetter(1))
                    order = [name[0][0] for name in bgOrder]
                    # plot bg
                    coffea.hist.plot1d(bg, ax=ax, stack=True, overflow="none", fill_opts=dict(color=[col for col in histAttributes]), order=order, clear=False)
                    hep.set_style("CMS")
                    hep.cms.label(llabel="Work in progress", lumi=np.round(self.config_inst.get_aux("lumi") / 1000.0, 2), loc=0, ax=ax)
                    if self.unblinded:
                        # normally, we have only two kinds of data, if more, adapt
                        dat = dataHists[0].add(dataHists[1])
                        data = dat.group("process", coffea.hist.Cat("process", "data"), {"data": ["data electron", "data muon"]})
                        coffea.hist.plot1d(data, uncStyle=dataUncStyle, ax=ax, clear=False)
                        coffea.hist.plotratio(data.sum("process"), bg.sum("process"), ax=rax, uncStyle=dataUncStyle, denom_fill_opts={}, guide_opts={}, unc="num", clear=False)
                        rax.set_ylabel("Ratio")
                        rax.set_ylim(0, 2)
                    # declare naming
                    leg = ax.legend(title="{0}: {1}".format(cat, var.x_title), ncol=1, loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
                    if self.logScale:
                        ax.set_yscale("log")
                        ax.set_ylim(0.0001, 1e12)
                    ax.set_xlabel(var.get_full_x_title())
                    ax.set_ylabel(var.get_full_y_title())
                    ax.autoscale(axis="x", tight=True)
                    if self.unblinded:
                        rax.set_xlabel(var.get_full_x_title())
                        rax.set_ylabel("Ratio")
                        rax.autoscale(axis="x", tight=True)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    ax.cla()
                    if self.unblinded:
                        rax.cla()
                    plt.close(fig)
            print("\n", " ---- Created {} pages ----".format(pdf.get_pagecount()), "\n")


class ArrayPlotting(CoffeaTask):
    density = luigi.BoolParameter(default=False)

    def requires(self):
        return {sel: CoffeaProcessor.req(self, lepton_selection=sel, workflow="local") for sel in ["Muon"]}  # , "Electron"

    def output(self):
        ending = ".png"
        if self.density:
            ending = "_density" + ending
        return {
            var: {
                "nominal": self.local_target(var + ending),
                "log": self.local_target(var + "_log" + ending),
            }
            for var in self.config_inst.variables.names()
        }

    def store_parts(self):
        return super(ArrayPlotting, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        inDict = self.input()  # ["collection"].targets
        # making clear which index belongs to which variable
        varNames = self.config_inst.variables.names()
        # create dir
        print(varNames)
        for var in tqdm(self.config_inst.variables):
            # new canvas for each variable
            fig, ax = plt.subplots(figsize=(18, 10))
            hep.set_style("CMS")
            hep.cms.label(llabel="Work in progress", loc=0, ax=ax)
            # iterate over the indices for each file
            feedback = []  # {}
            for key, value in inDict.items():
                np_dict = value["collection"].targets[0]
                # different key for each file, we ignore it for now, only interested in values
                # define empty hists
                np_hist = np.array([])
                np_0b = np.load(np_dict.path)  # np.load(np_dict["job_{}_N0b".format(i)].path)
                np_hist = np.append(np_hist, np_0b[:, varNames.index(var.name)])
                # integrate hist
                bins = np.arange(var.binning[1], var.binning[2], (var.binning[2] - var.binning[1]) / var.binning[0])
                back = np.sum(np.histogram(np_hist, bins=bins)[0])
                plt.hist(np_hist, bins=bins, histtype="step", label=key + ": {}".format(back), density=self.density)
                # feedback.update({key:np.sum(back[0])})
                feedback.append(back)
            # sorting the labels/handels of the plt hist by descending magnitude of integral
            order = np.argsort((-1) * np.array(feedback))
            print(var.name, feedback)
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend(np.array(handles)[order], np.array(labels)[order])
            ax.set_xlabel(var.get_full_x_title())
            ax.set_ylabel(var.get_full_y_title())
            self.output()[var.name]["nominal"].parent.touch()
            plt.savefig(self.output()[var.name]["nominal"].path)
            ax.set_yscale("log")
            plt.savefig(self.output()[var.name]["log"].path)
            plt.gcf().clear()


class DNNHistoryPlotting(DNNTask):
    """
    opening history callback and plotting curves for training
    """

    def requires(self):
        return PytorchMulticlass.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout, batch_size=self.batch_size, learning_rate=self.learning_rate)

    def output(self):
        return {"loss_plot": self.local_target("torch_loss_plot.png"), "acc_plot": self.local_target("torch_acc_plot.png")}

    def store_parts(self):
        # make plots for each use case
        return super(DNNHistoryPlotting, self).store_parts() + (self.analysis_choice,) + (self.n_nodes,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self):
        # retrieve history callback for trainings
        accuracy_stats = self.input()[0]["collection"].targets[0]["accuracy_stats"].load()
        loss_stats = self.input()[0]["collection"].targets[0]["loss_stats"].load()

        # read in values, skip first for val since Trainer does a validation step beforehand
        train_loss = loss_stats["train"]
        val_loss = loss_stats["val"]

        train_acc = accuracy_stats["train"]
        val_acc = accuracy_stats["val"]

        self.output()["loss_plot"].parent.touch()
        plt.plot(
            np.arange(0, len(val_loss), 1),
            val_loss,
            label="loss on valid data",
            color="orange",
        )
        plt.plot(
            np.arange(1, len(train_loss) + 1, 1),
            train_loss,
            label="loss on train data",
            color="green",
        )
        plt.legend()
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.savefig(self.output()["loss_plot"].path)
        plt.gcf().clear()

        plt.plot(
            np.arange(0, len(val_acc), 1),
            val_acc,
            label="acc on vali data",
            color="orange",
        )
        plt.plot(
            np.arange(1, len(train_acc) + 1, 1),
            train_acc,
            label="acc on train data",
            color="green",
        )
        plt.legend()
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        plt.savefig(self.output()["acc_plot"].path)
        plt.gcf().clear()


class DNNEvaluationPlotting(DNNTask):
    normalize = luigi.Parameter(default="true", description="if confusion matrix gets normalized")

    def requires(self):
        return dict(
            data=ArrayNormalisation.req(self),
            model=PytorchMulticlass.req(
                self,
                n_layers=self.n_layers,
                n_nodes=self.n_nodes,
                dropout=self.dropout,
                debug=False,
            ),
        )
        # return DNNTrainer.req(
        #    self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout
        # )

    def output(self):
        return {
            "ROC": self.local_target("pytorch_ROC.png"),
            "confusion_matrix": self.local_target("pytorch_confusion_matrix.png"),
        }

    def store_parts(self):
        # make plots for each use case
        return (
            super(DNNEvaluationPlotting, self).store_parts()
            + (self.analysis_choice,)
            # + (self.channel,)
            # + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
        )

    @law.decorator.timeit(publish_message=True)
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self):
        # from IPython import embed;embed()

        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template").keys())
        all_processes = list(self.config_inst.get_aux("DNN_process_template").keys())

        path = self.input()["model"]["collection"].targets[0]["model"].path

        # load complete model
        reconstructed_model = torch.load(path)

        # load all the prepared data thingies
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()

        test_dataset = util.ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(y_test))

        # val_loss, val_acc = reconstructed_model.evaluate(X_test, y_test)
        # print("Test accuracy:", val_acc)

        y_predictions = []
        with torch.no_grad():
            reconstructed_model.eval()
            for X_test_batch, y_test_batch in test_loader:
                y_test_pred = reconstructed_model(X_test_batch)

                y_predictions.append(y_test_pred.numpy())

            # test_predict = reconstructed_model.predict(X_test)
            y_predictions = np.array(y_predictions[0])
            test_predictions = np.argmax(y_predictions, axis=1)

            # "signal"...
            predict_signal = np.array(y_predictions)[:, 1]

        self.output()["confusion_matrix"].parent.touch()

        # Roc curve, compare labels and predicted labels
        fpr, tpr, tresholds = sk.metrics.roc_curve(y_test[:, 1], predict_signal)

        plt.plot(
            fpr,
            tpr,
            label="AUC: {0}".format(np.around(sk.metrics.auc(fpr, tpr), decimals=3)),
        )
        plt.plot([0, 1], [0, 1], ls="--")
        plt.xlabel(" fpr ", fontsize=16)
        plt.ylabel("tpr", fontsize=16)
        plt.title("ROC", fontsize=16)
        plt.legend()
        plt.savefig(self.output()["ROC"].path)
        plt.gcf().clear()

        # from IPython import embed;embed()
        # Correlation Matrix Plot
        # plot correlation matrix
        pred_matrix = sk.metrics.confusion_matrix(
            np.argmax(y_test, axis=-1),
            test_predictions,  # np.concatenate(test_predictions),
            normalize=self.normalize,
        )

        print(pred_matrix)
        # TODO
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # cax = ax.matshow(pred_matrix, vmin=-1, vmax=1)
        cax = ax.imshow(pred_matrix, vmin=0, vmax=1, cmap="plasma")
        fig.colorbar(cax)
        for i in range(n_processes):
            for j in range(n_processes):
                text = ax.text(
                    j,
                    i,
                    np.round(pred_matrix[i, j], 3),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14,
                )
        ticks = np.arange(0, n_processes, 1)
        # Let the horizontal axes labeling appear on bottom
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(all_processes)
        ax.set_yticklabels(all_processes)
        ax.set_xlabel("Predicted Processes")
        ax.set_ylabel("Real Processes")
        # ax.grid(linestyle="--", alpha=0.5)
        plt.savefig(self.output()["confusion_matrix"].path)
        plt.gcf().clear()


"""
Plot DNN distribution on test set
"""


class DNNDistributionPlotting(DNNTask):
    def requires(self):
        return dict(
            data=ArrayNormalisation.req(self),
            model=PytorchMulticlass.req(
                self,
                n_layers=self.n_layers,
                n_nodes=self.n_nodes,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
                debug=False,
            ),
        )

    def output(self):
        return {
            "real_processes": self.local_target("DNN_distribution.png"),
            "predicted": self.local_target("DNN_predicted_distribution.png"),
            "score_by_group": self.local_target("DNN_score_by_process.pdf"),
            "2D_variable_plots": self.local_target("DNN_score_2D_variables.pdf"),
        }

    def store_parts(self):
        # make plots for each use case
        return super(DNNDistributionPlotting, self).store_parts() + (self.analysis_choice,) + (self.channel,) + (self.n_layers,) + (self.n_nodes,) + (self.dropout,) + (self.batch_size,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self):
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template").keys())
        all_processes = list(self.config_inst.get_aux("DNN_process_template").keys())

        # load all the prepared data thingies
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()

        # val_loss, val_acc = reconstructed_model.evaluate(X_test, y_test)
        # print("Test accuracy:", val_acc)
        test_dataset = util.ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(y_test))
        path = self.input()["model"]["collection"].targets[0]["model"].path

        # load complete model
        reconstructed_model = torch.load(path)
        """
        y_predictions = []
        with torch.no_grad():

            reconstructed_model.eval()
            for X_test_batch, y_test_batch in test_loader:

                y_test_pred = reconstructed_model(X_test_batch)

                y_predictions.append(y_test_pred.numpy())

            # test_predict = reconstructed_model.predict(X_test)
            y_predictions = np.array(y_predictions[0])
            test_predictions = np.argmax(y_predictions, axis=1)
        """
        test_predict = reconstructed_model(torch.tensor(X_test))
        # FIXME
        test_predict = reconstructed_model.softmax(test_predict)

        self.output()["real_processes"].parent.touch()

        colors = ["black", "blue", "red", "yellow", "brown"]

        ### label
        fig = plt.figure()

        for i in range(n_processes):
            plt.hist(
                test_predict.detach().numpy()[y_test[:, i] == 1][:, i],
                label=all_processes[i],
                histtype="step",
                density=True,
                linewidth=1.5,
                color=colors[i],
                bins=10,
            )

        plt.xlabel("DNN Score", fontsize=16)
        plt.ylabel("Score", fontsize=16)
        plt.title("Real processes", fontsize=16)
        plt.legend()
        plt.savefig(self.output()["real_processes"].path)
        plt.gcf().clear()

        ### argmax
        fig = plt.figure()

        for i in range(n_processes):
            plt.hist(
                test_predict.detach().numpy()[np.argmax(test_predict.detach().numpy(), axis=1) == i][:, i],
                label=all_processes[i],
                histtype="step",
                density=True,
                linewidth=1.5,
                color=colors[i],
                bins=10,
            )

        plt.xlabel("DNN Score", fontsize=16)
        plt.ylabel("Score", fontsize=16)
        plt.title("Predicted processes", fontsize=16)
        plt.legend()
        plt.savefig(self.output()["predicted"].path)
        plt.gcf().clear()

        # create pdf object to save figures on separate pages
        with PdfPages(self.output()["score_by_group"].path) as pdf:
            for i, proc in enumerate(all_processes):
                fig = plt.figure()

                for j in range(n_processes):
                    plt.hist(
                        test_predict.detach().numpy()[y_test[:, i] == 1][:, j],
                        label=all_processes[j] + " score",
                        histtype="step",
                        density=True,
                        linewidth=1.5,
                        color=colors[j],
                        bins=10,
                    )
                plt.xlabel("DNN Score", fontsize=16)
                plt.ylabel("Score", fontsize=16)
                plt.title("DNN Scores for {}".format(proc), fontsize=16)
                plt.legend()

                pdf.savefig(fig)
                plt.close(fig)

        with PdfPages(self.output()["2D_variable_plots"].path) as pdf:
            for i, proc in enumerate(all_processes):
                for j, var in enumerate(self.config_inst.variables.names()):
                    fig = plt.figure()

                    plt.hist2d(
                        test_predict.detach().numpy()[y_test[:, i] == 1][:, i],
                        X_test[:, j][y_test[:, i] == 1],
                        bins=50,
                    )

                    plt.xlabel("DNN Score", fontsize=16)
                    plt.ylabel(
                        "{} {}".format(
                            self.config_inst.get_variable(var).x_title,
                            self.config_inst.get_variable(var).unit,
                        ),
                        fontsize=16,
                    )
                    plt.title("2D {} {}".format(proc, var), fontsize=16)
                    # plt.legend()
                    plt.colorbar()

                    pdf.savefig(fig)
                    plt.close(fig)


class PlotFeatureImportance(DNNTask):  # , HTCondorWorkflow, law.local
    def requires(self):
        return dict(
            data=ArrayNormalisation.req(self),
            model=PytorchMulticlass.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout),
        )

    def output(self):
        return {key: self.local_target("feature_importance.png".replace(".png", "_process_{}.png".format(key))) for key in self.config_inst.get_aux("DNN_process_template").keys()}

    def store_parts(self):
        # make plots for each use case
        return super(PlotFeatureImportance, self).store_parts() + (self.analysis_choice,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    def create_branch_map(self):
        # overwrite branch map
        n = 1
        return list(range(n))

        # return {i: i for i in range(n)}

    # Helper method to print importances and visualize distribution
    # def visualize_importances(self, feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):

    def run(self):
        # load complete model
        path = self.input()["model"]["collection"].targets[0]["model"].path
        reconstructed_model = torch.load(path)

        # load all the prepared data thingies
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()
        test_input_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
        baseline = torch.zeros(2, 14)

        ig = IntegratedGradients(reconstructed_model)

        out_probs = reconstructed_model(test_input_tensor).detach().numpy()
        out_classes = np.argmax(out_probs, axis=1)
        test_labels = np.argmax(y_test, axis=1)

        test_input_tensor.requires_grad_()
        print("Test Accuracy:", sum(out_classes == test_labels) / len(test_labels))

        feature_names = self.config_inst.variables.names()

        # FIXME, should be nothardcoded but dependent on some processes
        for j, key in enumerate(self.config_inst.get_aux("DNN_process_template").keys()):
            attr, delta = ig.attribute(test_input_tensor[:2000], target=j, return_convergence_delta=True)
            attr = attr.detach().numpy()

            # print('IG Attributions:', attr)
            # print('Convergence Delta:', delta)

            importances = np.mean(attr, axis=0)

            # plotting
            title = "Average Feature Importances for process {}".format(key)
            axis_title = "Features"
            # ugly to do it each time, but least amount of code
            self.output()[key].parent.touch()
            for i in range(len(feature_names)):
                print(feature_names[i], ": ", "%.3f" % (importances[i]))
            x_pos = np.arange(len(feature_names))
            # if plot:
            plt.figure(figsize=(12, 8))
            plt.bar(x_pos, importances, align="center")
            plt.xticks(x_pos, feature_names, wrap=True, rotation=30, rotation_mode="anchor")
            plt.xlabel(axis_title)
            plt.title(title)
            plt.savefig(self.output()[key].path)


class PlotNeuronConductance(DNNTask):  # , HTCondorWorkflow, law.local
    def requires(self):
        return dict(
            data=ArrayNormalisation.req(self),
            model=PytorchMulticlass.req(self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout),
        )

    def output(self):
        return self.local_target("Neurons.png")

    def store_parts(self):
        # make plots for each use case
        return super(PlotNeuronConductance, self).store_parts() + (self.analysis_choice,) + (self.dropout,) + (self.batch_size,) + (self.learning_rate,)

    def run(self):
        # load complete models
        path = self.input()["model"]["collection"].targets[0]["model"].path
        reconstructed_model = torch.load(path)

        # load all the prepared data thingies
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()
        test_input_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
        test_input_tensor.requires_grad_()

        feature_names = self.config_inst.variables.names()

        # also plot layer Conductance
        cond = LayerConductance(reconstructed_model, reconstructed_model.layer_1)
        cond_vals = cond.attribute(test_input_tensor[:1000], target=0)
        cond_vals = cond_vals.detach().numpy()

        plt.figure(figsize=(12, 8))
        title = "Average Neuron Importances"
        axis_title = "Neurons"
        x_pos = np.arange(len(feature_names))
        # from IPython import embed;embed()
        plt.bar(range(64), np.mean(cond_vals, axis=0), align="center")
        # plt.xticks(range(64), feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        self.output().parent.touch()
        plt.savefig(self.output().path)

        for i in range(64):
            plt.figure()
            plt.hist(cond_vals[:, i], 100)
            plt.title("Neuron {} Distribution".format(i))
            plt.savefig(self.output().path.replace(".png", "_neuron{}.png".format(i)))
            plt.close()