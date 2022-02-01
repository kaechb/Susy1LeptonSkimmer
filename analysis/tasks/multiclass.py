import law
import numpy as np
import pickle
import luigi.util
from luigi import BoolParameter, IntParameter, FloatParameter, ChoiceParameter
from law.target.collection import TargetCollection
import tensorflow as tf
from tensorflow import keras
import sklearn as sk
import sklearn.model_selection as skm
from rich.console import Console

from tasks.basetasks import DNNTask, HTCondorWorkflow
from tasks.arraypreparation import ArrayNormalisation

"""
DNN Stuff
"""

law.contrib.load("tensorflow")


class DNNTrainer(DNNTask, HTCondorWorkflow, law.LocalWorkflow):

    use_sqrt = BoolParameter(default=False)
    norm_class_weights = FloatParameter(default=1)
    job_number = IntParameter(
        default=1, description="how many HTCondo jobs are started"
    )

    def create_branch_map(self):
        # overwrite branch map
        n = self.job_number
        return list(range(n))

        # return {i: i for i in range(n)}

    def requires(self):
        # return PrepareDNN.req(self)
        return ArrayNormalisation.req(self, channel="N0b_CR")

    def output(self):
        return {
            "saved_model": self.local_target("saved_model"),
            "history_callback": self.local_target("history.pckl"),
            # test acc for optimizationdata for plotting
            "test_acc": self.local_target("test_acc.json"),
        }

    def store_parts(self):
        # debug_str = ''
        if self.debug:
            debug_str = "debug"
        else:
            debug_str = ""

        # put hyperparameters in path to make an easy optimization search
        return (
            super(DNNTrainer, self).store_parts()
            + (self.analysis_choice,)
            + (self.channel,)
            + (self.n_layers,)
            + (self.n_nodes,)
            + (self.dropout,)
            + (self.batch_size,)
            + (self.learning_rate,)
            + (debug_str,)
        )

    def build_model_sequential(self, n_variables, n_processes):
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

    def build_model_functional(
        self, n_variables, n_processes, n_layers=3, n_nodes=256, dropout=0.2
    ):
        # build a fully conncected DNN from defined paramters

        inp = keras.Input((n_variables,))
        x = keras.layers.BatchNormalization(
            axis=1, trainable=False, input_shape=(n_variables,)
        )(inp)

        for i in range(n_layers):
            x = keras.layers.Dense(
                n_nodes, activation=tf.nn.relu, kernel_regularizer="l2"
            )(x)
            x = keras.layers.Dropout(dropout)(x)
            x = keras.layers.BatchNormalization()(x)

        out = keras.layers.Dense(n_processes, activation=tf.nn.softmax)(x)
        model = keras.models.Model(inputs=inp, outputs=out)

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )
        return model

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

    def data_generator(self, X_train, y_train, batch_size, n_processes):

        # generate a batch for each step of the epoch, currently drawing random

        while True:

            arrays, one_hot_labels = [], []
            individual_batch_size = batch_size // n_processes

            for i in range(n_processes):
                arr = X_train[y_train[:, i] == 1]
                choice = np.random.choice(
                    np.arange(0, len(arr), 1), size=individual_batch_size, replace=False
                )
                arrays.append(arr[choice])
                labels = np.zeros((individual_batch_size, n_processes))
                labels[:, i] = 1
                one_hot_labels.append(labels)

            yield np.concatenate(arrays), np.concatenate(one_hot_labels)
            # print("yielded!")

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):

        # TENSORBOARD_PATH = (
        # self.output()["saved_model"].dirname
        # + "/logs/fit/"
        # + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # )

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
        gamma = np.ones((n_variables,))
        beta = np.zeros((n_variables,))

        # initiliazemodel and set first layer
        model = self.build_model_functional(
            n_variables=n_variables,
            n_processes=n_processes,
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            dropout=self.dropout,
        )
        # model_seq = self.build_model_sequential(n_variables=n_variables, n_processes=n_processes)
        model.layers[1].set_weights([gamma, beta, means, stds])

        # display model summary
        model.summary()

        # define callbacks to be used during the training
        # tensorboard = keras.callbacks.TensorBoard(
        # log_dir=TENSORBOARD_PATH, histogram_freq=0, write_graph=True
        # )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=25, min_lr=0.001
        )
        stop_of = keras.callbacks.EarlyStopping(
            monitor="val_loss",  # accuracy
            verbose=1,
            min_delta=0.0,
            patience=50,
            restore_best_weights=True,  # for now
        )

        # calc class weights for training so all classes get some love
        # scale up by abundance, additional factor to tone the values down a little
        # class_weights = self.calc_class_weights(
        #    y_train, norm=self.norm_class_weights, sqrt=self.use_sqrt
        # )
        # print("\nclass weights", class_weights, "\n")

        # check if everything looks fines, exit by crtl+D
        if self.debug:
            from IPython import embed

            embed()

        # plot schematic model graph
        self.output()["history_callback"].parent.touch()
        keras.utils.plot_model(
            #    model, to_file=self.output()["saved_model"].path + "/dnn_graph.png"
            model,
            to_file=self.output()["history_callback"].parent.path + "/dnn_scheme.png",
        )

        history_callback = model.fit(
            self.data_generator(X_train, y_train, self.batch_size, n_processes),
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            verbose=2,
            # iterate enough for smallest dataset
            steps_per_epoch=len(y_train[y_train[:, 1] == 1]) // self.batch_size,
            callbacks=[stop_of, reduce_lr],  # tensorboard],
        )

        # call without data generator
        """
        history_callback = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=max_epochs,
            verbose=2,
            callbacks=[stop_of],# , reduce_lr],  # tensorboard],
            # catch uneven input distributions
            class_weight=class_weights,
            # sample_weight=np.array(train_weights),
        )
        """

        # save model
        self.output()["saved_model"].parent.touch()
        model.save(self.output()["saved_model"].path)

        # save callback for plotting
        with open(self.output()["history_callback"].path, "wb") as f:
            pickle.dump(history_callback.history, f)

        console = Console()
        # load test dta/labels and evaluate on unseen data
        test_loss, test_acc = model.evaluate(X_test, y_test)
        console.print(
            "\n[u][bold magenta]Test accuracy on channel {}:[/bold magenta][/u]".format(
                self.channel
            )
        )
        console.print(test_acc, "\n")
        self.output()["test_acc"].dump({"test_acc": test_acc})


class DNNHyperParameterOpt(DNNTask, HTCondorWorkflow, law.LocalWorkflow):
    x = 1

    def create_branch_map(self):
        # overwrite branch map
        n = 1
        return list(range(n))

    def requires(self):
        # require DNN trainer with different configurations and monitor performance
        layers = [2]  # ,3,4]
        nodes = [128, 256]  # ,256,512]
        dropout = [0.2]  # ,0.4]
        grid_search = {
            "{}_{}_{}".format(lay, nod, drop): DNNTrainer.req(
                self,
                n_layers=lay,
                n_nodes=nod,
                dropout=drop,
                epochs=5,
                workflow="local",
                # job_number=len(layers)*len(nodes)*len (dropout),
                # workers=len(layers)*len(nodes)*len(dropout),
            )
            for lay in layers
            for nod in nodes
            for drop in dropout
        }

        return grid_search

    def output(self):
        return self.local_target("performance.json")
        # return self.local_target("x_{}_{}.json".format(self.iteration, self.branch))

        """
        idea akin to example, iterate trough branches and call for each configuration the DNN Trainer, then somehow evaluate
        """

    def store_parts(self):

        return super(DNNHyperParameterOpt, self).store_parts() + (self.analysis_choice,)

    def run(self):
        from IPython import embed

        embed()

        # output = yield Objective.req(self, x=x, iteration=self.branch, branch=-1)
        # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, BayesSearchCV

        performance = {}

        for key in self.input().keys():
            test_acc = self.input()[key]["test_acc"].load()["test_acc"]
            # ["collection"].targets[0]
            performance.update({key: test_acc})

        self.output().dump(performance)


class PrintOptDNNConfig(DNNTask):
    def requires(self):
        return DNNHyperParameterOpt.req(self)

    def output(self):
        return self.local_target("config.json")

    def store_parts(self):
        return super(PrintOptDNNConfig, self).store_parts() + (self.analysis_choice,)

    def run(self):
        # from IPython import embed; embed()

        performance = self.input()["collection"].targets[0].load()
        best_key = max(performance, key=performance.get)

        parameters = ["layers", "nodes", "dropout"]

        console = Console()
        console.print("\n[u][bold magenta]Best parameters are:[/bold magenta][/u]")
        for i, para in enumerate(parameters):
            print(para, best_key.split("_")[i])

        # print("best performance with layers:{layers} nodes:{nodes} dropout:{dropout}".format())
