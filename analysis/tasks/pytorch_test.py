import law
import numpy as np
import pickle
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
import pytorch_lightning as pl

# captum
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

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
            + (self.n_nodes,)
            + (self.dropout,)
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
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=-1)
        _, y_test_tags = torch.max(y_test, dim=-1)

        correct_pred = (y_pred_tags == y_test_tags).float()
        acc = correct_pred.sum() / len(correct_pred)
        # from IPython import embed;embed()
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

        class_weights = self.calc_class_weights(y_train)

        # declare device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

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

        self.steps_per_epoch = (
            n_processes * np.sum(y_test[:, 0] == 1) // self.batch_size
        )

        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            # sampler=util.EventBatchSampler(
            #   y_train, self.batch_size, n_processes, self.steps_per_epoch,
            #   ),
            num_workers=8,
        )

        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=10 * self.batch_size,  # , shuffle=True  # len(val_dataset
            num_workers=8,
        )  # =1

        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=10 * self.batch_size,
            num_workers=8,  # , shuffle=True  # self.batch_size
        )

        """
        at some point, we have to include k fold cross validation
        https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=k_folds, shuffle=True)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        """

        # declare lighnting callbacks
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode="max",
        )

        # have to apply softmax somewhere on validation/inference FIXME

        # declare model
        model = util.MulticlassClassification(
            num_feature=n_variables,
            num_class=n_processes,
            means=means,
            stds=stds,
            dropout=self.dropout,
            class_weights=torch.tensor(list(class_weights.values())),
            n_nodes=self.n_nodes,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        print(model)
        # accuracy_stats = {"train": [], "val": []}
        # loss_stats = {"train": [], "val": []}

        # Trainer, for gpu gpus=1
        trainer = pl.Trainer(max_epochs=self.epochs, num_nodes=1)

        if self.debug:
            from IPython import embed

            embed()
        # fit the trainer, includes the whole training loop
        trainer.fit(model, train_dataloader, val_dataloader)

        """
        for e in tqdm(range(1, self.epochs + 1)):

            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for i, (X_train_batch, y_train_batch) in enumerate(train_loader):

                #if i > self.steps_per_epoch:
                #    break
                # squeeze extra dimension coming from who knows:
                X_train_batch, y_train_batch = X_train_batch.squeeze(
                    0
                ), y_train_batch.squeeze(0)

                optimizer.zero_grad()
                y_train_pred = model(X_train_batch)

                train_loss = criterion(y_train_pred, y_train_batch)

                train_acc = self.multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
                # print("check", train_epoch_acc, train_acc)

            print("trained", time() - tic)

            # VALIDATION
            with torch.no_grad():

                val_epoch_loss = 0
                val_epoch_acc = 0

                model.eval()
                for X_val_batch, y_val_batch in val_loader:

                    X_val_batch, y_val_batch = X_val_batch.squeeze(
                    0
                    ), y_val_batch.squeeze(0)


                    y_val_pred = model(X_val_batch)

                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = self.multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            # have to rethink how to best reweight val / train loss, the data loader is different
            loss_stats["train"].append(
                train_epoch_loss /self.steps_per_epoch
            )  # len(train_loader)
            loss_stats["val"].append(val_epoch_loss / len(val_loader))
            accuracy_stats["train"].append(
                train_epoch_acc / self.steps_per_epoch
            )  # len(train_loader)
            accuracy_stats["val"].append(val_epoch_acc /len(val_loader))

            #from IPython import embed;embed()

            print(
                "Epoch: {epoch} | Train loss: {train_loss} | Val loss: {val_loss} | Train Acc: {train_acc} | Val Acc: {val_acc}".format(
                    epoch=e,
                    train_loss=np.round(
                        train_epoch_loss / self.steps_per_epoch, 4
                    ),  # / len(train_loader)
                    val_loss=np.round(val_epoch_loss / len(val_loader),4),
                    train_acc=np.round(train_epoch_acc / self.steps_per_epoch, 3),  #
                    val_acc=np.round(val_epoch_acc / len(val_loader),3),
                )
            )
        """

        # evaluate test set
        with torch.no_grad():

            test_epoch_loss = 0
            test_epoch_acc = 0

            model.eval()
            for X_test_batch, y_test_batch in test_loader:

                X_test_batch, y_test_batch = X_test_batch.squeeze(
                    0
                ), y_test_batch.squeeze(0)

                y_test_pred = model(X_test_batch)

                test_loss = criterion(y_test_pred, y_test_batch)
                test_acc = self.multi_acc(y_test_pred, y_test_batch)

                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()

        # print result
        console = Console()
        console.print(
            "\n[u][bold magenta]Test accuracy on channel {}:[/bold magenta][/u]".format(
                self.channel
            )
        )
        console.print(test_acc, "\n")

        # save away all stats
        self.output()["model"].touch()
        torch.save(model, self.output()["model"].path)
        self.output()["loss_stats"].dump(model.loss_stats)
        self.output()["accuracy_stats"].dump(model.accuracy_stats)


class FeatureImportance(DNNTask):
    def requires(self):
        return (
            PytorchMulticlass.req(
                self, n_layers=self.n_layers, n_nodes=self.n_nodes, dropout=self.dropout
            ),
        )

    def run(self):

        path = self.input()[0]["collection"].targets[0]["model"].path

        reconstructed_model = torch.load(path)

        ig = IntegratedGradients(reconstructed_model)

        input = torch.rand(2, 14)
        baseline = torch.zeros(2, 14)

        attributions, delta = ig.attribute(
            input, baseline, target=0, return_convergence_delta=True
        )

        print("IG Attributions:", attributions)
        print("Convergence Delta:", delta)

        from IPython import embed

        embed()
