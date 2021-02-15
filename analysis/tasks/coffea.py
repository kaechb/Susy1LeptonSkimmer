# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
from luigi import BoolParameter, Parameter
from coffea.processor import ProcessorABC
import json
import time
import numpy as np

# other modules
from tasks.basetasks import AnalysisTask
from utils.CoffeaBase import *
from tasks.makefiles import WriteFileset

class CoffeaProcessor(AnalysisTask):
    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter()

    def requires(self):
        return WriteFileset.req(self)

    #def load_corrections(self):
    #    return {corr: target.load() for corr, target in self.input()["corrections"].items()}

    def output(self):
        return self.local_target("array.npy")

    def store_parts(self):
        parts = (self.analysis_choice, self.processor)
        if self.debug:
            parts += ("debug", self.Processor.debug_dataset)
        return super(CoffeaProcessor, self).store_parts() + parts

    def run(self):

        with open(self.input().path, "r") as read_file:
            fileset=json.load(read_file)

        # fileset = {
            # (dsname, shift): [target.path for target in files.targets if target.exists()]
            # for (dsname, shift), files in fileset.items()
            # if self.config_inst.datasets.has(dsname)
            # and (processor_inst.dataset_shifts or shift == "nominal")
        # }
        # # remove empty datasets
        # empty = [key for key, paths in fileset.items() if not paths]
        # if empty:
            # for e in empty:
                # del fileset[e]
            # logger.warning("skipping empty datasets: %s", ", ".join(map(str, sorted(empty))))

        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter()
        if self.processor == "Histograner":
            processor_inst = Histogramer()

        tic = time.time()

        #, metrics
        out = processor.run_uproot_job(
            fileset,
            treename="nominal",
            processor_instance=processor_inst,
            #pre_executor=processor.futures_executor,
            #pre_args=dict(workers=32),
            executor=processor.iterative_executor,
            #executor_args=dict(
                #nano=True,
                #savemetrics=1,
                # xrootdtimeout=30,
                # align_clusters=True,
                # processor_compression=None,
                #**ea,
            #),
            chunksize=100000,
        )
        toc = time.time()
        print(np.round(toc - tic,2), "s")

        #from IPython import embed;embed()

        # save outputs
        self.output().parent.touch()
        path=self.output().path
        np.save(path, out["arrays"]["hl"])
