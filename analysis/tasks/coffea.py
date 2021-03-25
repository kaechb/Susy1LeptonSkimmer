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

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from utils.CoffeaBase import *
from tasks.makefiles import WriteFileset


class CoffeaProcessor(
    DatasetTask, HTCondorWorkflow, law.LocalWorkflow
):  # AnalysisTask):
    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter()
    debug_dataset = "TTJets_sl_fromt"

    """
    this is a HTCOndor workflow, normally it will get submitted with configurations defined
    in the htcondor_bottstrap.sh or the basetasks.HTCondorWorkflow
    If you want to run this locally, just use --workflow local in the terminal
    """

    def __init__(self, *args, **kwargs):
        super(CoffeaProcessor, self).__init__(*args, **kwargs)

    def requires(self):
        return WriteFileset.req(self)

    # def load_corrections(self):
    # return {corr: target.load() for corr, target in
    # self.input()["corrections"].items()}

    def output(self):
        out = "array_"  # "array.npy"
        if self.processor == "Histogramer":
            out = "hists.coffea"
        return self.local_target(out)

    def store_parts(self):
        parts = (self.analysis_choice, self.processor)
        if self.debug:
            parts += ("debug", self.debug_dataset)
        return super(CoffeaProcessor, self).store_parts() + parts

    def run(self):
        with open(self.input().path, "r") as read_file:
            fileset = json.load(read_file)

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
            processor_inst = ArrayExporter(self)
        if self.processor == "Histogramer":
            processor_inst = Histogramer(self)

        tic = time.time()

        if self.debug:
            # from IPython import embed;embed()
            fileset = {self.debug_dataset: [fileset[self.debug_dataset][0]]}

        # , metrics
        out = processor.run_uproot_job(
            fileset,
            treename="nominal",
            processor_instance=processor_inst,
            # pre_executor=processor.futures_executor,
            # pre_args=dict(workers=32),
            executor=processor.iterative_executor,
            # executor_args=dict(savemetrics=1,
            # schema=BaseSchema,),
            chunksize=100000,
        )
        # executor_args=dict(
        # nano=True,
        # savemetrics=True,
        # xrootdtimeout=30,
        # align_clusters=True,
        # processor_compression=None,
        # **ea,
        # ),

        # show summary
        console = Console()
        all_events = out["n_events"]["sum_all_events"]
        toc = time.time()
        # print(np.round(toc - tic, 2), "s")

        # from IPython import embed;embed()
        total_time = toc - tic
        console.print("\n[u][bold magenta]Summary metrics:[/bold magenta][/u]")
        console.print(f"* Total time: {total_time:.2f}s")
        console.print(f"* Total events: {all_events:e}")
        console.print(f"* Events / s: {all_events/total_time:.0f}")

        # save outputs
        self.output().parent.touch()
        path = self.output().path

        if self.processor == "ArrayExporter":
            for cat in out["arrays"]:
                np.save(path + "{}.npy".format(cat), out["arrays"][cat]["hl"])

        if self.processor == "Histogramer":
            self.output().dump(out["histograms"])
