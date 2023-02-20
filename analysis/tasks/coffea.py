# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
from luigi import BoolParameter, Parameter, IntParameter
from coffea import processor
from coffea.nanoevents import TreeMakerSchema, BaseSchema, NanoAODSchema
import json
import time
import numpy as np
import uproot as up
from rich.console import Console
from tqdm import tqdm

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from utils.CoffeaBase import *
from tasks.makefiles import WriteFileset, WriteDatasets


class CoffeaTask(DatasetTask):
    """
    token task to define attributes # FIXME: remove absolute paths
    """
    processor = Parameter(default="ArrayExporter")
    debug = BoolParameter(default=False)
    debug_dataset = Parameter(default="data_mu_C")  # take a small set to reduce computing time
    debug_str = Parameter(default="/nfs/dust/cms/user/wiens/CMSSW/CMSSW_12_1_0/Testing/2022_11_10/TTJets/TTJets_1.root")
    file = Parameter(default="/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_12_1_0/Batch/2022_11_24/2017/Data/root/SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1_NANOAOD_1.0.root")
    job_number = IntParameter(default=1)
    data_key = Parameter(default="SingleMuon")
    # parameter with selection we use coffea
    lepton_selection = Parameter(default="Muon")

class CoffeaProcessor(CoffeaTask, HTCondorWorkflow, law.LocalWorkflow):

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
        return WriteDatasets.req(self)
        # return WriteFileset.req(self)

    def create_branch_map(self):
        # if self.debug:
        return list(range(1))
        # return list(range(self.job_dict[self.data_key]))  # self.job_number

    def output(self):
        return self.local_target("signal.npy")

    def store_parts(self):
        parts = (self.analysis_choice,self.processor,self.lepton_selection,)
        return super(CoffeaProcessor, self).store_parts() + parts

    @law.decorator.timeit(publish_message=True)
    def run(self):
        # FIXME mom
        data_dict = self.input()["dataset_dict"].load()  # ["SingleMuon"]  # {self.dataset: [self.file]}
        dataPath = self.input()["dataset_path"].load()
        # declare processor 
        if self.processor == "ArrayExporter":
            processor_inst = ArrayExporter(self, Lepton=self.lepton_selection)
        if self.processor == "Histogramer":
            processor_inst = Histogramer(self, self.lepton_selection)
        # building together the respective strings to use for the coffea call
        treename = self.lepton_selection
        key_name = list(data_dict.keys())[0]
        subset = sorted(data_dict[key_name])
        dataset = key_name.split("_")[0]
        start =  time.time()
        with up.open(dataPath + "/" + subset[self.branch]) as file:
            primaryDataset = "MC" 
        isData = file["MetaData"]["IsData"].array()[0]
        fileset = {
            dataset: {"files": [dataPath + "/" + subset[self.branch]],"metadata": {"PD": primaryDataset,"IsData": isData,},}        
            }
        if self.debug:
            with up.open(dataPath + "/" + subset[self.branch]) as file:
                primaryDataset = file["MetaData"]["primaryDataset"].array()[0]
            fileset = {
                dataset: {"files": [dataPath + "/" + subset[self.branch]],"metadata": {"PD": primaryDataset},}
            }

        # call imported processor, magic happens here
        out = processor.run_uproot_job(fileset,treename=treename,processor_instance=processor_inst,executor=processor.iterative_executor,
                                       executor_args=dict(status=False), chunksize=10000)
        # show summary
        console = Console()
        all_events = out["n_events"]["sum_all_events"]
        total_time =  time.time() - start
        console.print("\n[u][bold magenta]Summary metrics:[/bold magenta][/u]")
        console.print(f"* Total time: {total_time:.2f}s")
        console.print(f"* Total events: {all_events:e}")
        console.print(f"* Events / s: {all_events/total_time:.0f}")
        # save outputs, seperated for processor, both need different touch calls
        if self.processor == "ArrayExporter":
            self.output().parent.touch()
            for cat in out["arrays"]:
                self.output().dump(out["arrays"][cat]["hl"].value)
                # self.output()[cat].dump(out["arrays"][cat]["hl"].value)
            # hacky way of defining if task is done FIXME
            # self.output().dump(np.array([1]))
        if self.processor == "Histogramer":
            self.output().parent.touch()
            self.output().dump(out["histograms"])


class SubmitCoffeaPerDataset(CoffeaTask):  # , HTCondorWorkflow , law.LocalWorkflow

    def requires(self):
        return WriteDatasets.req(self)

    def output(self):
        # return self.local_target("joblist_{}.json".format(self.branch))
        return self.local_target("joblist.json")

    def store_parts(self):
        return super(SubmitCoffeaPerDataset, self).store_parts() + (
            self.analysis_choice,
            self.processor,
            # self.dataset,
            self.file.split("/")[-1].replace(".root", ""),
        )

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        print("\nIn Submit\n")
        # goal of this task is to fill a dict with all the coffea locations
        joblist = {}
        testDict = self.input()["dataset_dict"].load() 
        dataPath = self.input()["dataset_path"].load()
        jobNumberDict = self.input()["job_number_dict"].load()
        # doing a loop for each small file on the naf
        for key in self.job_dict.keys():
            for sel in ["Muon", "Electron"]:
                cof_proc = CoffeaProcessor.req(self,processor=self.processor,data_key=key, lepton_selection=sel,)
                # find output of the coffea processor
                outTarget = cof_proc.localize_output().args[0]["collection"].targets
                newTarget = {}
                # unpack Localfiletargers, since json dump wont work otherwise
                if self.processor == "ArrayExporter":
                    for path in outTarget[0].keys():
                        # FIXME
                        newTarget[path] = outTarget[0][path].path
                joblist.update({key + "_" + sel: newTarget})
                # and lets submit this job
                # run = cof_proc.run()
                print("running branch for file:", key, sel, cof_proc)
                # self.output().dump(joblist)
                test = yield cof_proc
        self.output().dump(joblist)

class CollectCoffeaOutput(CoffeaTask):

    def requires(self):
        return {"{}_{}".format(sel, dat): 
                CoffeaProcessor.req(self,data_key=dat,lepton_selection=sel,job_number=self.job_dict[dat])
            for sel in ["Muon", "Electron"] for dat in ["SingleMuon", "MET", "SingleElectron"]
        }

    def output(self):
        return self.local_target("event_counts.json")

    def store_parts(self):
        return super(CollectCoffeaOutput, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        inDict = self.input()  
        # making clear which index belongs to which variable
        varNames = self.config_inst.variables.names()
        print(varNames)
        # signal_events = 0
        eventCounts = {}
        # iterate over the indices for each file
        for key, value in inDict.items():
            tot_events, signal_events = 0, 0
            np_dict = value["collection"].targets[0]
            # different key for each file, we ignore it for now, only interested in values
            for i in tqdm(range(len(np_dict) // 2)):
                np_0b = np.load(np_dict["job_{}_N0b".format(i)].path)
                Dphi = np_0b[:, varNames.index("Dphi")]
                LT = np_0b[:, varNames.index("LT")]
                HT = np_0b[:, varNames.index("HT")]
                n_jets = np_0b[:, varNames.index("n_jets")]
                # at some point, we have to define the signal regions
                LT1 = (LT > 250) & (LT < 450) & (Dphi > 1) & (HT > 500)
                LT2 = (LT > 450) & (LT < 650)& (Dphi > 0.75) & (HT > 500)
                LT3 = (LT > 650) & (Dphi > 0.75) & (HT > 500)
                LT1_nj5 = np_0b[LT1 & (n_jets == 5)]
                LT1_nj67 = np_0b[LT1 & (n_jets > 5) & (n_jets < 8)]
                LT1_nj8i = np_0b[LT1 & (n_jets > 7)]
                LT2_nj5 = np_0b[LT2 & (n_jets == 5)]
                LT2_nj67 = np_0b[LT2 & (n_jets > 5) & (n_jets < 8)]
                LT2_nj8i = np_0b[LT2 & (n_jets > 7)]
                LT3_nj5 = np_0b[LT3 & (n_jets == 5)]
                LT3_nj67 = np_0b[LT3 & (n_jets > 5) & (n_jets < 8)]
                LT3_nj8i = np_0b[LT3 & (n_jets > 7)]
                signal_events += (len(LT1_nj5) + len(LT1_nj67) + len(LT1_nj8i) + len(LT2_nj5) + len(LT2_nj67) + len(LT2_nj8i) + len(LT3_nj5) + len(LT3_nj67) + len(LT3_nj8i))
                tot_events += len(np_0b)
            count_dict = {
                key: {"tot_events": tot_events, "signal_events": signal_events}
            }
            print(count_dict)
            eventCounts.update(count_dict)
        self.output().dump(eventCounts)

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
        
    def store_parts(self):
        return super(GroupCoffeaProcesses, self).store_parts() + (self.analysis_choice,)

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        import uproot3 as up3
        hists = self.input()["collection"][0].load()
        datasets = self.config_inst.datasets.names()
        categories = self.config_inst.categories.names()
        # create dir and write coffea to root hists
        self.output().parent.touch()
        with up3.recreate(self.output().path) as root_file:
            histVariables = hists.keys()
            for var in histVariables:
                for dat in datasets:
                    for cat in categories:
                        histName = self.template.format(variable=var, process=dat, category=cat)
                        if "data" in dat:
                            histName = self.template.format(variable=var, process="data_obs", category=cat)
                        root_file[histName] = coffea.hist.export1d(hists[var][(dat, cat)].project(var))
