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
from rich.console import Console
import numpy as np
import boost_histogram as bh
import aghast
import uproot
import ROOT


import logging

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from utils.CoffeaBase import *
from tasks.makefiles import WriteFileset

logger = logging.getLogger(__name__)

"""
Just import everything and try out stuff
"""


class TestDummy(DatasetTask, HTCondorWorkflow, law.LocalWorkflow):  # AnalysisTask):
    debug = BoolParameter(default=True)
    random_number = IntParameter(default=42)
    x = luigi.ListParameter(default=[1])
    iteration = luigi.IntParameter(default=1)

    def create_branch_map(self):
        return {i: x for i, x in enumerate(self.x)}

    def output(self):
        return self.local_target("fileset_{}_{}.json".format(self.random_number, self.branch))

    def run(self):
        # make the output directory

        print("\ndeeedledoo\n")
        arr = np.random.normal(size=1_000_000)
        import time
        from IPython import embed

        embed()
        time.sleep(5)
        self.output().parent.touch()

        # unchanged syntax
        # test_file = "/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"

        fileset = {self.random_number: "haha"}

        """
        fileset = {}

        for dat in self.config_inst.datasets:
            fileset.update(
                {
                    dat.name: dat.keys,
                }
            )

        print(fileset)
        """

        # from IPython import embed;embed()

        with open(self.output().path, "w") as file:
            json.dump(fileset, file)


class TestSubmitter(DatasetTask, law.LocalWorkflow):
    def requires(self):
        # randoms = np.random.randint(0,100, 3)
        randoms = [1, 2, 3]
        rand_dict = {}
        for number in randoms:
            rand_dict.update({number: TestDummy.req(self, random_number=int(number), workflow="local")})

        return rand_dict

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
