# coding: utf-8

import logging
import os
import law
import order as od
import law.contrib.coffea
import luigi
import coffea
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
from tqdm.auto import tqdm


# other modules
from tasks.basetasks import *
from tasks.coffea import CoffeaProcessor


class PlotCoffeaHists(ConfigTask):

    log_scale = luigi.BoolParameter()
    unblinded = luigi.BoolParameter()
    scale_signal = luigi.IntParameter(default=1)

    def requires(self):
        return CoffeaProcessor.req(
            self, processor="Histogramer"
        )  # , workflow="local", branch=-1)

    def output(self):
        path = ""
        if self.log_scale:
            path += "_log"
        if self.unblinded:
            path += "_data"
        return self.local_target("hists{}.pdf".format(path))

    def run(self):
        inp = self.input()["collection"][0].load()
        self.output().parent.touch()
        # create pdf object to save figures on separate pages
        with PdfPages(self.output().path) as pdf:

            # print dummy hist, because first one is broken
            # does not work either FIXME

            # plot each hist
            for var in tqdm(self.config_inst.variables, unit="variable"):

                hists = inp[var.name]
                categories = [h.name for h in hists.identifiers("category")]

                for cat in categories:

                    bg_hists = []
                    hists_attr = []

                    # ax = fig.add_subplot(number,1,i+1)
                    fig = plt.figure()
                    ax = fig.add_subplot()

                    # ax.set_title("{0}: {1}".format(cat, var.name))
                    ax.autoscale(axis="x", tight=True)

                    for proc in self.config_inst.processes:
                        #from IPython import embed;embed()
                        #if "QCD" in proc.name:

                        # for each process, map childs together to plot in one, get rid of cat axis
                        child_hists = hists[
                            [p[0].name for p in proc.walk_processes()], cat
                        ]
                        mapping = {
                            proc.label: [p[0].name for p in proc.walk_processes()]
                        }
                        # a = child_hists.sum("dataset", overflow='none')
                        # from IPython import embed;embed()

                        # bg_hists.append()
                        # hist_attr.append([proc.label, proc.color])
                        # a=child_hists.sum("dataset", overflow='none').project(var.name)
                        # a.axes()[0].label = proc.label
                        grouped = child_hists.group(
                            "dataset",
                            coffea.hist.Cat("process", proc.label),
                            mapping,
                        ).integrate("category")

                        bg_hists.append(grouped)
                        hists_attr.append(proc.color)

                        # if more datasets: FIXME
                        # FIXME axes kinda empty? integrate gets rid of one

                        # for dat in hists.identifiers("dataset"):
                    #from IPython import embed;embed()
                    bg = bg_hists[0]
                    for i in range(1, len(bg_hists)):
                        bg.add(bg_hists[i])

                    coffea.hist.plot1d(
                        # inp[var.name].integrate("category"),
                        bg,
                        ax=ax,stack=True,
                        overflow="none",
                        fill_opts=dict(color=[col for col in hists_attr]),
                        #fill_opts=hists_attr,   #dict(color=proc.color),
                        # legend_opts=dict(proc.label)
                        clear=False,
                        # overlay="dataset",
                    )
                    hep.set_style("CMS")
                    hep.cms.label(
                        llabel="Work in progress",
                        lumi=np.round(self.config_inst.get_aux("lumi") / 1000.0, 2),
                        loc=0,
                        ax=ax,
                    )

                    # declare naming
                    leg = ax.legend(
                        title="{0}: {1}".format(cat, var.x_title),
                        ncol=1,
                        loc="upper right",
                        bbox_to_anchor=(1.04, 1),
                        borderaxespad=0,
                    )

                    if self.log_scale:
                        ax.set_yscale("log")
                        ax.set_ylim(0.0001, 1e9)

                    ax.set_xlabel(var.get_full_x_title())
                    ax.set_ylabel(var.get_full_y_title())

                    plt.tight_layout()
                    pdf.savefig(fig)
                    ax.cla()
                    fig.clear()

            print("\n", " ---- Created {} pages ----".format(pdf.get_pagecount()), "\n")


class ArrayPlotting(ConfigTask):
    def requires(self):
        return CoffeaProcessor.req(
            self, processor="ArrayExporter", debug=True, workflow="local"
        )  # , workflow="local", branch=-1)

    def output(self):
        return self.local_target("hists")

    def run(self):

        # inp = self.input()["collection"][0].load()
        array = self.input()["collection"].targets[0]["N0b_TTZ_qq"].load()
        # path=self.input()["collection"].targets[0].path

        print(self.config_inst.variables.names(), ":")
        print(array)
        from IPython import embed

        embed()
