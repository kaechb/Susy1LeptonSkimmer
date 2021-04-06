# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
from luigi import BoolParameter, Parameter
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


class TestDummy(DatasetTask, law.LocalWorkflow, HTCondorWorkflow):  # AnalysisTask):
    debug = BoolParameter(default=True)

    def output(self):
        return self.local_target("fileset.json")

    def run(self):
        # make the output directory

        print("deeedledoo")
        from IPython import embed

        embed()
        self.output().parent.touch()

        # unchanged syntax
        # test_file = "/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"

        fileset = {}

        for dat in self.config_inst.datasets:
            fileset.update(
                {
                    dat.name: dat.keys,
                }
            )

        print(fileset)

        # from IPython import embed;embed()

        with open(self.output().path, "w") as file:
            json.dump(fileset, file)
