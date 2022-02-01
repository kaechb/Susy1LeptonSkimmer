from law.util import make_list
import os

"""
Helper class for writing datacards. Whereever you want to use this class,
you have to ensure a valid CH installation.
"""


class DatacardWriter(object):
    def __init__(self, ch, analysis, mass="125", era="2017"):

        self._cb = ch.CombineHarvester()

        self.analysis = analysis
        self._mass = mass
        self._era = era

    @property
    def cb(self):
        return self._cb

    @property
    def mass(self):
        return self._mass

    @property
    def era(self):
        return self._era

    def add_observation(self, channel, category):
        mass = make_list(self.mass)
        analysis = make_list(self.analysis)
        era = make_list(self.era)
        channel = make_list(channel)
        category = make_list(category)
        print(
            "Add observations with mass {}, analysis {}, era {}, channel {} and category {}.".format(
                mass, analysis, era, channel, category
            )
        )
        self.cb.AddObservations(mass, analysis, era, channel, category)

    def add_processes(self, channel, process, category, is_signal):
        mass = make_list(self.mass)
        analysis = make_list(self.analysis)
        era = make_list(self.era)
        channel = make_list(channel)
        category = make_list(category)
        print(
            "Add {} with mass {}, analysis {}, era {}, channel {}, process {} and category {}.".format(
                "signals" if is_signal else "backgrounds",
                mass,
                analysis,
                era,
                channel,
                process,
                category,
            )
        )
        self.cb.AddProcesses(mass, analysis, era, channel, process, category, is_signal)

    def add_signals(self, channel, process, category):
        self.add_processes(channel, process, category, True)

    def add_backgrounds(self, channel, process, category):
        self.add_processes(channel, process, category, False)

    def add_shape_systematic(self, name, strength, channel, process, ch):
        channel = make_list(channel)
        process = make_list(process)
        self.cb.cp().channel(channel).process(process).AddSyst(
            self.cb, name, "shape", ch.SystMap()(strength)
        )

    def remove_shape_uncertainties(self):
        # Filter all systematics of type shape.
        self.cb.FilterSysts(lambda systematic: systematic.type() == "shape")
        # There are also systematics, which can have a mixed type of lnN/shape, where CH returns only lnN as type. Such which values 1.0 and 0.0 are assumed to be shape uncertainties.
        self.cb.FilterSysts(
            lambda systematic: (systematic.value_u() == 1.0)
            and (systematic.value_d() == 0.0)
        )

    def add_normalization_systematic(self, name, strength, channel, process, ch):
        channel = make_list(channel)
        process = make_list(process)
        self.cb.cp().channel(channel).process(process).AddSyst(
            self.cb, name, "lnN", ch.SystMap()(strength)
        )

    def add_bin_by_bin_uncertainties(
        self, processes, ch, add_threshold=0.1, merge_threshold=0.5, fix_norm=True
    ):
        bin_by_bin_factory = ch.BinByBinFactory()
        bin_by_bin_factory.SetAddThreshold(add_threshold)
        bin_by_bin_factory.SetMergeThreshold(merge_threshold)
        bin_by_bin_factory.SetFixNorm(fix_norm)
        bin_by_bin_factory.MergeBinErrors(self.cb.cp().process(processes))
        bin_by_bin_factory.AddBinByBin(self.cb.cp().process(processes), self.cb)
        self.cb.SetGroup("bbb", [".*_bin_\\d+"])
        self.cb.SetGroup("syst_plus_bbb", [".*"])

    def scale_expectation(
        self, scale_factor, no_norm_rate_bkg=False, no_norm_rate_sig=False
    ):
        self.cb.cp().backgrounds().ForEachProc(
            lambda process: process.set_rate(
                (process.no_norm_rate() if no_norm_rate_bkg else process.rate())
                * scale_factor
            )
        )
        self.cb.cp().signals().ForEachProc(
            lambda process: process.set_rate(
                (process.no_norm_rate() if no_norm_rate_sig else process.rate())
                * scale_factor
            )
        )

    def scale_processes(self, scale_factor, processes, no_norm_rate=False):
        self.cb.cp().process(processes).ForEachProc(
            lambda process: process.set_rate(
                (process.no_norm_rate() if no_norm_rate else process.rate())
                * scale_factor
            )
        )

    def replace_observation_by_asimov_dataset(
        self, signal_mass=None, signal_processes=None
    ):
        def _replace_observation_by_asimov_dataset(observation):
            cb = (
                self.cb.cp()
                .analysis([observation.analysis()])
                .era([observation.era()])
                .channel([observation.channel()])
                .bin([observation.bin()])
            )
            background = cb.cp().backgrounds()

            signal = cb.cp().signals()
            if signal_mass:
                if signal_processes:
                    signal = (
                        cb.cp().signals().process(signal_processes).mass([signal_mass])
                    )
                else:
                    signal = cb.cp().signals().mass([signal_mass])
            elif signal_processes:
                signal = cb.cp().signals().process(signal_processes)

            observation.set_shape(background.GetShape() + signal.GetShape(), True)
            observation.set_rate(background.GetRate() + signal.GetRate())

        self.cb.cp().ForEachObs(_replace_observation_by_asimov_dataset)

    def auto_rebin(self, ch, threshold=0.0, unc_frac=0.9, mode=1):
        rebin = ch.AutoRebin()
        rebin.SetBinThreshold(threshold)
        rebin.SetBinUncertFraction(unc_frac)
        rebin.SetRebinMode(mode)
        rebin.SetPerformRebin(True)
        rebin.SetVerbosity(1)
        rebin.Rebin(self.cb, self.cb)

    def fix_negative_bins(self):
        def _fix_negative_bins(process):
            hist = process.ShapeAsTH1F()
            for i in range(hist.GetNbinsX()):
                if hist.GetBinContent(i) < 0.0:
                    print(
                        "Fixing negative bins for process {} in bin {}".format(
                            process.process(), i
                        )
                    )
                    hist.SetBinContent(i, 0.0)

        def _fix_negative_bins_sys(syst):
            # first fix shift down
            hist = syst.ShapeDAsTH1F()
            for i in range(hist.GetNbinsX()):
                if hist.GetBinContent(i) < 0.0:
                    print(
                        "Fixing negative bins for syst {} (shift down) in bin {}".format(
                            syst.name(), i
                        )
                    )
                    hist.SetBinContent(i, 0.0)
            # then fix shift up
            hist = syst.ShapeUAsTH1F()
            for i in range(hist.GetNbinsX()):
                if hist.GetBinContent(i) < 0.0:
                    print(
                        "Fixing negative bins for syst {} (shift up) in bin {}".format(
                            syst.name(), i
                        )
                    )
                    hist.SetBinContent(i, 0.0)

        # for each process
        self.cb.ForEachProc(_fix_negative_bins)
        # for each systematic
        self.cb.ForEachSyst(_fix_negative_bins_sys)

    def print_datacard(self):
        self.cb.PrintAll()

    def write_datacards(
        self, datacard_filename_template, root_filename_template, output_directory, ch
    ):
        # http://cms-analysis.github.io/CombineHarvester/classch_1_1_card_writer.html#details
        writer = ch.CardWriter(
            os.path.join("$TAG", datacard_filename_template),
            os.path.join("$TAG", root_filename_template),
        )
        writer.SetVerbosity(1)

        # enable writing datacards in cases where the mass does not have its original meaning
        if (len(self.cb.mass_set()) == 1) and (self.cb.mass_set()[0] == "*"):
            writer.SetWildcardMasses([])

        return writer.WriteCards(output_directory, self.cb)
