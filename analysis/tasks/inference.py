# coding: utf-8

import law
from law.util import make_list
import luigi

# from coffea.util import load

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from tasks.coffea import GroupCoffeaProcesses

from utils.sandbox import CMSSWSandboxTask


class DatacardProducer(DatasetTask): 
    "old example task to show datacard producing in task"
    SignalBin = luigi.Parameter(default="ab_1234", description="bin to plot")
    variable = luigi.Parameter(default="MET",description="variable to fit",)

    def requires(self):
        # check if we need root or arrays or coffea hists, root for now...
        return GroupCoffeaProcesses.req(self)

    def output(self):
        return {
            "datacard": self.local_target("datacard.txt"),
            "shapes": self.local_target("datacard_shapes.root"),
        }

    def store_parts(self):
        return (
            super(DatacardProducer, self).store_parts()
            + (self.analysis_choice,)
            + (self.SignalBin,)
            + (self.variable,)
        )

    def make_pairs(self, x):
        return [(i, c) for i, c in enumerate(x)]

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        import sys

        # FIXME: Paths
        print("Importing from /nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_10_2_13/src/CombineHarvester/CombineTools/python3")
        sys.path.append("/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_10_2_13/src/CombineHarvester/CombineTools/python3")
        import ch
        from utils.datacard import DatacardWriter

        # either import the datacard writer or heritage from him
        channel = self.SignalBin.split("_")[0]
        categories = self.make_pairs([self.config_inst.categories.names()])
        dw = DatacardWriter(ch=ch, analysis=self.analysis_choice, mass="125", era="2016")
        allProcesses = self.config_inst.get_aux("process_groups")["default"]
        for process in allProcesses:
            if "data" in process:
                allProcesses.remove(process)
        signal_processes = [self.config_inst.get_aux("signal_process")]
        background_processes = [proc for proc in allProcesses if str(self.config_inst.get_aux("signal_process")) not in proc]
        dw.add_observation(channel, categories)
        dw.add_signals(channel, signal_processes, categories)
        dw.add_backgrounds(channel, background_processes, categories)
        # import analysis specific uncertainity rates and add the to dw
        uncRatesAdder = self.analysis_import("recipes.uncertainityrates")
        uncRatesAdder.add_dw_uncertainities(dw=dw, channel=channel, ch=ch, allProcesses=allProcesses)
        for process in allProcesses:
            dw.cb.cp().process(make_list(process)).ExtractShapes(self.input()["root"].path, str(self.variable) + "_$PROCESS_$BIN", str(self.variable) + "_$PROCESS_$BIN_$SYSTEMATIC",)
        dw.replace_observation_by_asimov_dataset(self.category)
        # Perform auto-rebinning
        dw.auto_rebin(ch=ch)
        dw.fix_negative_bins()
        # print summary:
        with self.publish_step(law.util.colored("Datacard summary:", color="light_cyan")):
            self.publish_message("> Analyses: {}".format(dw.cb.analysis_set()))
            self.publish_message("> Eras: {}".format(dw.cb.era_set()))
            self.publish_message("> Masses: {}".format(dw.cb.mass_set()))
            self.publish_message("> Channels: {}".format(dw.cb.channel_set()))
            self.publish_message("> Processes: {}".format(dw.cb.process_set()))
            self.publish_message("> Signals: {}".format(dw.cb.cp().signals().process_set()))
            self.publish_message("> Backgrounds: {}".format(dw.cb.cp().backgrounds().process_set()))
            self.publish_message("> Categories: {}".format(dw.cb.bin_set()))

        # Write datacard
        dw.write_datacards(self.output()["datacard"].basename, self.output()["shapes"].basename, self.output()["datacard"].dirname, ch=ch)
        command = "sed -i -e '$a* autoMCStats 12' {}".format(self.output()["datacard"].path)
        law.util.interruptable_popen(command, shell=True, executable="/bin/bash")
