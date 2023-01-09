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
import uproot as up
from rich.console import Console

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from utils.CoffeaBase import *
from tasks.makefiles import WriteFileset, WriteDatasets


class CoffeaTask(DatasetTask):
    """
    token task to define attributes
    """

    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter(default=False)
    debug_dataset = Parameter(
        default="data_mu_C"
    )  # take a small set to reduce computing time
    debug_str = Parameter(
        default="/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root"
    )
    file = Parameter(
        default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root"
        # "/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root"
    )


class CoffeaProcessor(
    CoffeaTask, HTCondorWorkflow, law.LocalWorkflow
):  # AnalysisTask):

    """
    this is a HTCOndor workflow, normally it will get submitted with configurations defined
    in the htcondor_bottstrap.sh or the basetasks.HTCondorWorkflow
    If you want to run this locally, just use --workflow local in the command line
    Overall task to execute Coffea
    Config and actual code is found in utils
    """

    def __init__(self, *args, **kwargs):
        super(CoffeaProcessor, self).__init__(*args, **kwargs)

    def requires(self):
        return WriteFileset.req(self)

    # def load_corrections(self):
    # return {corr: target.load() for corr, target in
    # self.input()["corrections"].items()}

    def create_branch_map(self):
        return list(range(1))

    def output(self):
        datasets = self.config_inst.datasets.names()
        if self.debug:
            datasets = [self.debug_dataset]
        out = {
            cat + "_" + dat: self.local_target(cat + "_" + dat + ".npy")
            for dat in datasets
            for cat in self.config_inst.categories.names()
        }
        # overwrite array export logic if we want to histogram
        if self.processor == "Histogramer":
            out = self.local_target("hists.coffea")
        return out

    def store_parts(self):
        parts = (self.analysis_choice, self.processor)
        if self.debug:
            parts += (
                "debug",
                self.debug_dataset,
                self.debug_str.split("/")[-1].replace(".root", ""),
            )
        return super(CoffeaProcessor, self).store_parts() + parts

    # def scheduler_messages(self):
    # def empty():
    # return True
    # scheduler_messages.empty = empty
    # return scheduler_messages

    @law.decorator.timeit(publish_message=True)
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

        # declare professor
        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter(self)
        if self.processor == "Histogramer":
            processor_inst = Histogramer(self)

        tic = time.time()

        if self.debug:
            from IPython import embed

            # embed()
            # fileset = {self.debug_dataset: [fileset[self.debug_dataset][0]]}
            fileset = {self.debug_dataset: [self.debug_str]}
            # embed()

        # , metrics
        # call imported processor, magic happens here
        out = processor.run_uproot_job(
            fileset,
            treename="LeptonIncl",
            processor_instance=processor_inst,
            # pre_executor=processor.futures_executor,
            # pre_args=dict(workers=32),
            executor=processor.iterative_executor,
            executor_args=dict(status=False, desc="Trolling"),
            # schema=BaseSchema,),
            chunksize=10000,
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

        total_time = toc - tic
        console.print("\n[u][bold magenta]Summary metrics:[/bold magenta][/u]")
        console.print(f"* Total time: {total_time:.2f}s")
        console.print(f"* Total events: {all_events:e}")
        console.print(f"* Events / s: {all_events/total_time:.0f}")

        # save outputs
        # seperated for processor, both need different touch calls
        if self.processor == "ArrayExporter":
            self.output().popitem()[1].parent.touch()
            for cat in out["arrays"]:
                self.output()[cat].dump(out["arrays"][cat]["hl"].value)
            # hacky way of defining if task is done FIXME
            # self.output().dump(np.array([1]))

        if self.processor == "Histogramer":
            self.output().parent.touch()
            self.output().dump(out["histograms"])


class SubmitCoffeaPerDataset(CoffeaTask, HTCondorWorkflow, law.LocalWorkflow):
    def create_branch_map(self):
        # from IPython import embed; embed()
        # return a job for every dataset that has to be processed
        return list(range(2))

    def requires(self):
        return WriteDatasets.req(self)

    def output(self):
        return self.local_target("joblist.json")

    def store_parts(self):
        return super(SubmitCoffeaPerDataset, self).store_parts() + (
            self.analysis_choice,
            self.processor,
            self.dataset,
            self.file.split("/")[-1].replace(".root", ""),
        )

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        print("\nIn Submit\n")

        # goal of this task is to fill a dict with all the coffea locations
        # dataset = Parameter(default="data_e_C")

        joblist = {}

        # test_file = "/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root"
        # test_file = "/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root"

        # test_dict = {"TTJets_sl_fromt": [test_file]}
        test_dict = self.input()[
            "dataset_dict"
        ].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        data_path = self.input()["dataset_path"].load()

        # at some point, we have to select which dataset to process
        good_file_numbers = ["105", "106"]

        # doing a loop for each small file on the naf
        for key in test_dict.keys():
            if key != "SingleMuon":
                continue
            for file in test_dict[key]:
                # if not "105" in file and not "106" in file:
                if not good_file_numbers[self.branch]:
                    continue
                if not "Run2017C" in file:
                    continue
                print("loopdaloop")
                # defining how to convert dataset names
                data_dict = {"SingleMuon": "data_mu_C"}
                # define coffea Processor instance for this one dataset
                cof_proc = CoffeaProcessor.req(
                    self,
                    processor=self.processor,  # "ArrayExporter",
                    debug=True,
                    debug_dataset=data_dict[key],
                    debug_str=data_path + "/" + file,
                    # no_poll=True,  #just submit, do not initiate status polling
                    workflow="local",
                )
                # find output of the coffea processor
                # from IPython import embed; embed()
                # out_target = cof_proc.localize_output().args[0]["collection"].targets[0]
                out_target = cof_proc.localize_output().args[0]

                # unpack Localfiletargers, since json dump wont work otherwise
                if self.processor == "ArrayExporter":
                    for path in out_target.keys():
                        out_target[path] = out_target[path].path

                if self.processor == "Histogramer":
                    out_target = out_target.path

                joblist.update({key + "_" + file: out_target})

                # generates new graph at runtime
                # test = yield cof_proc

                # and lets submit this job
                # run = cof_proc.run()
                # from IPython import embed; embed()
                test = yield cof_proc
                self.output().dump(joblist)

        # with open(self.output().path, "w") as file:
        #    json.dump(joblist, file)


class CollectCoffeaOutput(CoffeaTask):
    def requires(self):
        return SubmitCoffeaPerDataset.req(
            self, dataset=self.debug_dataset, processor=self.processor
        )

    # def output(self):

    def store_parts(self):
        return super(GroupCoffeaProcesses, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        from IPython import embed

        embed()

        print(self.input())
        a = self.input().load()
        b = a[
            "data_e_C_/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root"
        ]["N1b_SR_data_e_C"]
        c = np.load(b)


class GroupCoffeaProcesses(DatasetTask):

    """
    Task to group coffea hist together if needed (e.g. get rid of an axis)
    Or reproduce root files for Combine
    """

    # histogram naming template:
    template = "{variable}_{process}_{category}"

    def requires(self):
        return CoffeaProcessor.req(self, processor="Histogramer")

    def output(self):
        return self.local_target("legacy_hists.root")
        # return dict(
        #    coffea=self.local_target("legacy_hists.coffea"),
        #    root=self.local_target("legacy_hists.root"),
        # )

    def store_parts(self):
        return super(GroupCoffeaProcesses, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):

        import uproot3 as up3

        # needed for newtrees
        # hists = coffea.util.load(self.input().path)
        hists = self.input()["collection"][0].load()

        datasets = self.config_inst.datasets.names()
        categories = self.config_inst.categories.names()

        # create dir and write coffea to root hists
        self.output().parent.touch()
        with up3.recreate(self.output().path) as root_file:
            var_keys = hists.keys()
            # var1='METPt'
            for var in var_keys:
                for dat in datasets:
                    for cat in categories:

                        # hacky way to get the hidden array of dict values
                        # arr=list(hists[var1][('TTJets_sl_fromt','N1b_CR')].values())[0]

                        # root_file[categories[0]] = up3.newtree({self.template.format(variable=var1, process=datasets[0], category=categories[0]):np.float64})
                        # root_file[categories[0]].extend({self.template.format(variable=var1, process=datasets[0], category=categories[0]):arr})

                        hist_name = self.template.format(
                            variable=var, process=dat, category=cat
                        )
                        if "data" in dat:
                            hist_name = self.template.format(
                                variable=var, process="data_obs", category=cat
                            )

                        # from IPython import embed;embed()
                        root_file[hist_name] = coffea.hist.export1d(
                            hists[var][(dat, cat)].project(var)
                        )

                        # cutflow variable may have to be an exception
