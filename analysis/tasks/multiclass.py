import law
import numpy as np
import pickle
import luigi.util
from luigi import BoolParameter, IntParameter, FloatParameter, ChoiceParameter
from law.target.collection import TargetCollection
import tensorflow as tf
from tensorflow import keras
import sklearn.model_selection as skm
from rich.console import Console

from tasks.basetasks import ConfigTask, HTCondorWorkflow
from tasks.arraypreparation import ArrayNormalisation

"""
DNN Stuff
"""

law.contrib.load("tensorflow")


class DNNTrainer(ConfigTask, HTCondorWorkflow, law.LocalWorkflow):

    channel = luigi.Parameter(default="0b", description="channel to train on")
    epochs = IntParameter(default=100)
    batch_size = IntParameter(default=1000)
    learning_rate = IntParameter(default=0.01)

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
            "saved_model": self.local_target("saved_model"),
            "history_callback": self.local_target("history.pckl"),
            # test data for plotting
            "test_data": self.local_target("test_data.npy"),
            "test_labels": self.local_target("test_labels.npy"),
        }

    def store_parts(self):
        return (
            super(DNNTrainer, self).store_parts()
            + (self.analysis_choice,)
            + (self.channel,)
        )

    def build_model(self, n_variables, n_processes):

        # try selu or elu https://keras.io/api/layers/activations/?
        # simple sequential model to start
        model = keras.Sequential(
            [
                # normalise input and keep the values
                keras.layers.BatchNormalization(
                    axis=1, trainable=False, input_shape=(n_variables,)
                ),
                keras.layers.Dense(256, activation=tf.nn.elu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256, activation=tf.nn.elu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(256, activation=tf.nn.elu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(n_processes, activation=tf.nn.softmax),
            ]
        )

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

    def calc_norm_parameter(self, data):

        dat = np.swapaxes(data, 0, 1)
        means, stds = [], []
        for var in dat:
            means.append(var.mean())
            stds.append(var.std())

        return np.array(means), np.array(stds)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self):

        # TENSORBOARD_PATH = (
        # self.output()["saved_model"].dirname
        # + "/logs/fit/"
        # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # )

        # all_processes = self.config_inst.get_aux("process_groups")["default"]

        # as luigi ints with default?
        batch_size = self.batch_size
        max_epochs = self.epochs

        # load data
        n_variables = len(self.config_inst.variables)
        n_processes = len(self.config_inst.processes) - 2  # substract data

        """
        train_data = np.load(self.input()[self.channel]["train"]["data"].path)
        train_labels = np.load(self.input()[self.channel]["train"]["label"].path)
        train_weights = np.load(self.input()[self.channel]["train"]["weight"].path)
        val_data = np.load(self.input()[self.channel]["validation"]["data"].path)
        val_labels = np.load(self.input()[self.channel]["validation"]["label"].path)
        test_data = np.load(self.input()[self.channel]["test"]["data"].path)
        test_labels = np.load(self.input()[self.channel]["test"]["label"].path)
        data = np.append(train_data, val_data, axis=0)
        data = np.append(data, test_data, axis=0)
        """

        # kinda bad solution, but I want to keep all processes as separate npy files
        arr_list = []

        for key in self.input().keys():
            arr_list.append((self.input()[key].load()))

        # dont concatenate, parallel to each other
        arr_conc = np.concatenate(list(a for a in arr_list[:-1]))
        labels = arr_list[-1]

        labels = np.swapaxes(labels, 0, 1)

        # split up test set 95
        Trainset, X_test, Trainlabel, y_test = skm.train_test_split(
            arr_conc, labels, test_size=0.05, random_state=42
        )

        # train and validation set 80:20
        X_train, X_val, y_train, y_val = skm.train_test_split(
            Trainset, Trainlabel, test_size=0.2, random_state=42
        )

        # from IPython import embed;embed()

        # configure the norm layer. Give it mu/sigma, layer is frozen
        # gamma, beta are for linear activations, so set them to unity transfo
        means, stds = self.calc_norm_parameter(arr_conc)
        print(means.shape)
        gamma = np.ones((n_variables,))
        beta = np.zeros((n_variables,))

        # initiliazemodel and set first layer
        model = self.build_model(n_variables=n_variables, n_processes=n_processes)
        model.layers[0].set_weights([gamma, beta, means, stds])

        # display model summary
        model.summary()
        # plot schematic model graph

        keras.utils.plot_model(
            #    model, to_file=self.output()["saved_model"].path + "/dnn_graph.png"
            model,
            to_file=self.output()["history_callback"].parent.path
            + "/dnn_wolbn_scheme.png",
        )

        # tensorboard = keras.callbacks.TensorBoard(
        # log_dir=TENSORBOARD_PATH, histogram_freq=0, write_graph=True
        # )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
        )
        stop_of = keras.callbacks.EarlyStopping(
            monitor="accuracy",
            verbose=1,  # val_accuracy
            min_delta=0.0,
            patience=20,
            restore_best_weights=True,  # for now
        )

        # calc class weights for training so all classes get some love
        # scale up by abundance, additional factor to tone the values down a little
        class_weights = {}
        for i in range(n_processes):
            class_weights.update({i: len(y_train) / (5 * np.sum(y_train[:, i]))})
        print("class weights", class_weights)

        history_callback = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=max_epochs,
            verbose=2,
            callbacks=[stop_of, reduce_lr],  # tensorboard],
            # catch uneven input distributions
            class_weight=class_weights,
            # sample_weight=np.array(train_weights),
        )

        # save model
        self.output()["saved_model"].parent.touch()
        model.save(self.output()["saved_model"].path)

        # save callback for plotting
        with open(self.output()["history_callback"].path, "wb") as f:
            pickle.dump(history_callback.history, f)

        # test data
        self.output()["test_data"].dump(X_test)
        self.output()["test_labels"].dump(y_test)

        console = Console()
        # load test dta/labels and evaluate on unseen data
        test_loss, test_acc = model.evaluate(X_test, y_test)
        console.print(
            "\n[u][bold magenta]Test accuracy on channel {}:[/bold magenta][/u]".format(
                self.channel
            )
        )
        console.print(test_acc, "\n")
