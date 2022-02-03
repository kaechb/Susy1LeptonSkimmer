import law
import numpy as np
import pickle
import luigi.util
from luigi import BoolParameter, IntParameter, FloatParameter, ChoiceParameter
import sklearn as sk
import sklearn.model_selection as skm
from rich.console import Console
from tqdm.auto import tqdm
from time import time

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data  # import Dataset, DataLoader, WeightedRandomSampler

from tasks.basetasks import DNNTask, HTCondorWorkflow
from tasks.arraypreparation import ArrayNormalisation

import utils.PytorchHelp as util


class PytorchMulticlass(DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    def create_branch_map(self):
        # overwrite branch map
        n = 1
        return list(range(n))

        # return {i: i for i in range(n)}

    def requires(self):
        # return PrepareDNN.req(self)
        return ArrayNormalisation.req(self, channel="N0b_CR")

    def output(self):
        return {
            "model": self.local_target("model.pt"),
            "loss_stats": self.local_target("loss_stats.json"),
            "accuracy_stats": self.local_target("accuracy_stats.json"),
            # test acc for optimizationdata for plotting
            # "test_acc": self.local_target("test_acc.json"),
        }

    def store_parts(self):
        # debug_str = ''
        if self.debug:
            debug_str = "debug"
        else:
            debug_str = ""

        # put hyperparameters in path to make an easy optimization search
        return (
            super(PytorchMulticlass, self).store_parts()
            + (self.channel,)
            # + (self.n_layers,)
            # + (self.n_nodes,)
            # + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (debug_str,)
        )

    def calc_class_weights(self, y_train, norm=1, sqrt=False):
        # calc class weights to battle imbalance
        # norm to tune down huge factors, sqrt to smooth the distribution
        from sklearn.utils import class_weight

        weight_array = norm * class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(np.argmax(y_train, axis=-1)),
            y=np.argmax(y_train, axis=-1),
        )

        if sqrt:
            # smooth by exponential function
            # return dict(enumerate(np.sqrt(weight_array)))
            return dict(enumerate((weight_array) ** 0.88))
        if not sqrt:
            # set at minimum to 1.0
            # return dict(enumerate([a if a>1.0 else 1.0 for a in weight_array]))
            return dict(enumerate(weight_array))

    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        _, y_test_tags = torch.max(y_test, dim=1)

        correct_pred = (y_pred_tags == y_test_tags).float()
        acc = correct_pred.sum() / len(correct_pred)

        # acc = torch.round(acc * 100)

        return acc

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):

        tic = time()

        # define dimensions, working with aux template for processes
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.get_aux("DNN_process_template").keys())

        # load the prepared data and labels
        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()
        X_val = self.input()["X_val"].load()
        y_val = self.input()["y_val"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()

        # definition for the normalization layer
        means, stds = (
            self.input()["means_stds"].load()[0],
            self.input()["means_stds"].load()[1],
        )

        # class_weights = self.calc_class_weights(y_train)

        print(1, time() - tic)

        # datasets are loaded
        train_dataset = util.ClassifierDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        )
        val_dataset = util.ClassifierDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
        )
        test_dataset = util.ClassifierDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        )

        print(2, time() - tic)

        train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            # sampler=weighted_sampler
        )
        val_loader = data.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size
        )  # =1

        print(3, time() - tic)

        # declare model
        model = util.MulticlassClassification(
            num_feature=n_variables, num_class=n_processes, means=means, stds=stds
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        print(model)

        accuracy_stats = {"train": [], "val": []}
        loss_stats = {"train": [], "val": []}

        print(4, time() - tic)

        for e in tqdm(range(1, self.epochs + 1)):

            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0

            model.train()

            """
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)start.record()
            z = x + y
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            print(start.elapsed_time(end))
            """

            for i, (X_train_batch, y_train_batch) in enumerate(train_loader):

                if i > self.steps_per_epoch:
                    break

                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)

                train_loss = criterion(y_train_pred, y_train_batch)
                # from IPython import embed;embed()
                train_acc = self.multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            print("trained", time() - tic)

            # VALIDATION
            with torch.no_grad():

                val_epoch_loss = 0
                val_epoch_acc = 0

                model.eval()
                for X_val_batch, y_val_batch in val_loader:

                    y_val_pred = model(X_val_batch)

                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = self.multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats["train"].append(
                train_epoch_loss / self.batch_size * self.steps_per_epoch
            )  # len(train_loader)
            loss_stats["val"].append(val_epoch_loss / len(val_loader))
            accuracy_stats["train"].append(
                train_epoch_acc / self.batch_size * self.steps_per_epoch
            )  # len(train_loader)
            accuracy_stats["val"].append(val_epoch_acc / len(val_loader))

            print(
                "Epoch: {epoch} | Train loss: {train_loss} | Val loss: {val_loss} | Train Acc: {train_acc} | Val Acc: {val_acc}".format(
                    epoch=e,
                    train_loss=np.round(train_epoch_loss / len(train_loader), 4),
                    val_loss=np.round(val_epoch_loss / len(val_loader), 4),
                    train_acc=np.round(train_epoch_acc / len(train_loader), 3),
                    val_acc=np.round(val_epoch_acc / len(val_loader), 3),
                )
            )
            print("full_epoch", time() - tic)

        self.output()["model"].touch()
        torch.save(model, self.output()["model"].path)
        self.output()["loss_stats"].dump(loss_stats)
        self.output()["accuracy_stats"].dump(accuracy_stats)
