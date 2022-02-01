# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
import luigi
from luigi import BoolParameter, Parameter, IntParameter, ListParameter
from coffea import processor
from coffea.nanoevents import TreeMakerSchema, BaseSchema, NanoAODSchema
import json
import time
import numpy as np
from rich.console import Console

import logging

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from utils.CoffeaBase import *
from tasks.makefiles import WriteFileset

logger = logging.getLogger(__name__)

"""
taken from
https://github.com/riga/law/tree/master/examples/parallel_optimization
"""


class Optimizer(DatasetTask, law.LocalWorkflow):
    """
    Workflow that runs optimization.
    """

    iterations = IntParameter(default=10, description="Number of iterations")
    n_parallel = IntParameter(default=4, description="Number of parallel evaluations")
    n_initial_points = IntParameter(
        default=10,
        description="Number of random sampled values \
        before starting optimizations",
    )

    def create_branch_map(self):
        return list(range(self.iterations))

    def requires(self):
        if self.branch == 0:
            return None
        return Optimizer.req(self, branch=self.branch - 1)

    def output(self):
        return self.local_target("optimizer_{}.pkl".format(self.branch))

    def run(self):
        import skopt

        optimizer = (
            self.input().load()
            if self.branch != 0
            else skopt.Optimizer(
                dimensions=[skopt.space.Real(-5.0, 10.0), skopt.space.Real(0.0, 15.0)],
                random_state=1,
                n_initial_points=self.n_initial_points,
            )
        )

        x = optimizer.ask(n_points=self.n_parallel)

        output = yield Objective.req(self, x=x, iteration=self.branch, branch=-1)

        y = [f.load()["y"] for f in output["collection"].targets.values()]

        optimizer.tell(x, y)

        print(
            "minimum after {} iterations: {}".format(self.branch + 1, min(optimizer.yi))
        )

        with self.output().localize("w") as tmp:
            tmp.dump(optimizer)


# @luigi.util.inherits(Optimizer)
class OptimizerPlot(DatasetTask, law.LocalWorkflow):
    """
    Workflow that runs optimization and plots results.
    """

    iterations = IntParameter(default=10, description="Number of iterations")
    n_parallel = IntParameter(default=4, description="Number of parallel evaluations")
    n_initial_points = IntParameter(
        default=10,
        description="Number of random sampled values \
        before starting optimizations",
    )

    plot_objective = BoolParameter(
        default=True,
        description="Plot objective. \
        Can be expensive to evaluate for high dimensional input",
    )

    def create_branch_map(self):
        return list(range(self.iterations))

    def requires(self):
        return Optimizer.req(self)

    def has_fitted_model(self):
        return (
            self.plot_objective
            and (self.branch + 1) * self.n_parallel >= self.n_initial_points
        )

    def output(self):
        collection = {
            "evaluations": self.local_target("evaluations_{}.pdf".format(self.branch)),
            "convergence": self.local_target("convergence_{}.pdf".format(self.branch)),
        }

        if self.has_fitted_model():
            collection["objective"] = self.local_target(
                "objective_{}.pdf".format(self.branch)
            )

        return law.SiblingFileCollection(collection)

    def run(self):
        from skopt.plots import plot_objective, plot_evaluations, plot_convergence
        import matplotlib.pyplot as plt

        result = self.input().load().run(None, 0)
        output = self.output()

        with output.targets["convergence"].localize("w") as tmp:
            plot_convergence(result)
            tmp.dump(plt.gcf(), bbox_inches="tight")
        plt.close()
        with output.targets["evaluations"].localize("w") as tmp:
            plot_evaluations(result, bins=10)
            tmp.dump(plt.gcf(), bbox_inches="tight")
        plt.close()
        if self.has_fitted_model():
            plot_objective(result)
            with output.targets["objective"].localize("w") as tmp:
                tmp.dump(plt.gcf(), bbox_inches="tight")
            plt.close()


class Objective(DatasetTask, law.LocalWorkflow):

    """
    def requires(self):
        #randoms = np.random.randint(0,100, 3)
        randoms=[1,2,3]
        rand_dict = {}
        for number in randoms:
            rand_dict.update({number:TestDummy.req(self, random_number=int(number), workflow="local"})

        return rand_dict
    """

    x = ListParameter(default=[1, 2, 3])
    iteration = IntParameter()

    def create_branch_map(self):
        return {i: x for i, x in enumerate(self.x)}

    def output(self):
        return self.local_target("x_{}_{}.json".format(self.iteration, self.branch))

    def run(self):
        print("\ndone")
        print(self.input())

        from skopt.benchmarks import branin

        with self.output().localize("w") as tmp:
            tmp.dump({"x": self.branch_data, "y": branin(self.branch_data)})
