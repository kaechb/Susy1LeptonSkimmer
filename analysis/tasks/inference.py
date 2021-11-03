# coding: utf-8

import law
from law.util import make_list
import luigi

# from coffea.util import load

# other modules
from tasks.basetasks import DatasetTask, HTCondorWorkflow
from tasks.coffea import GroupCoffeaProcesses

class DatacardProducer(DatasetTask):  #, CMSSWSandboxTask):

    "old example task to show datacard producing in task"

    signal_bin = luigi.Parameter(default="ab_1234", description="bin to plot")
    variable = luigi.Parameter(default="MET", description="variable to fit",)

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
            + (self.signal_bin,)
            + (self.variable,)
        )

    def make_pairs(self, x):
        return [(i, c) for i, c in enumerate(x)]

    @law.decorator.timeit(publish_message=True)
    @law.decorator.safe_output
    def run(self):
        #import ROOT
        # normally, import combine using CMSSW
        #from CombineHarvester.CombineTools import ch
        import sys
        sys.path.append("/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_10_2_13/src/CombineHarvester/CombineTools/python")
        import ch

        from utils.datacard import DatacardWriter
        # either import the datacard writer or heritage from him

        channel = self.signal_bin.split("_")[0]

        categories = self.make_pairs([self.config_inst.categories.names()])

        from IPython import embed;embed()

        dw = DatacardWriter(
            ch=ch, analysis=self.analysis_choice, mass="125", era="2016"
        )

        all_processes = self.config_inst.get_aux("process_groups")["default"]
        for process in all_processes:
            if "data" in process:
                all_processes.remove(process)
        signal_processes = [self.config_inst.get_aux("signal_process")]
        background_processes = [
            proc
            for proc in all_processes
            if str(self.config_inst.get_aux("signal_process")) not in proc
        ]

        dw.add_observation(channel, categories)
        dw.add_signals(channel, signal_processes, categories)
        dw.add_backgrounds(channel, background_processes, categories)

        # import analysis specific uncertainity rates and add the to dw
        unc_rates_adder = self.analysis_import("recipes.uncertainityrates")
        unc_rates_adder.add_dw_uncertainities(
            dw=dw, channel=channel, ch=ch, all_processes=all_processes
        )

        for process in all_processes:
            dw.cb.cp().process(make_list(process)).ExtractShapes(
                self.input()["root"].path,
                str(self.variable) + "_$PROCESS_$BIN",
                str(self.variable) + "_$PROCESS_$BIN_$SYSTEMATIC",
                # "$SYSTEMATIC/$PROCESS/" + str(channel) + "/$BIN/" + str(self.variable),
            )

        dw.replace_observation_by_asimov_dataset(self.category)
        # dw.add_bin_by_bin_uncertainties(all_processes, ch=ch, add_threshold=0.1, merge_threshold=0.5, fix_norm=True)
        # Perform auto-rebinning
        dw.auto_rebin(ch=ch)
        dw.fix_negative_bins()

        # print summary:
        with self.publish_step(
            law.util.colored("Datacard summary:", color="light_cyan")
        ):
            self.publish_message("> Analyses: {}".format(dw.cb.analysis_set()))
            self.publish_message("> Eras: {}".format(dw.cb.era_set()))
            self.publish_message("> Masses: {}".format(dw.cb.mass_set()))
            self.publish_message("> Channels: {}".format(dw.cb.channel_set()))
            self.publish_message("> Processes: {}".format(dw.cb.process_set()))
            self.publish_message(
                "> Signals: {}".format(dw.cb.cp().signals().process_set())
            )
            self.publish_message(
                "> Backgrounds: {}".format(dw.cb.cp().backgrounds().process_set())
            )
            self.publish_message("> Categories: {}".format(dw.cb.bin_set()))

        # Write datacard
        dw.write_datacards(
            self.output()["datacard"].basename,
            self.output()["shapes"].basename,
            self.output()["datacard"].dirname,
            ch=ch,
        )
        command = "sed -i -e '$a* autoMCStats 12' {}".format(
            self.output()["datacard"].path
        )
        law.util.interruptable_popen(command, shell=True, executable="/bin/bash")
