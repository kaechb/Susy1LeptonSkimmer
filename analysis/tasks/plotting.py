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
        return CoffeaProcessor.req(self, processor="Histogramer")

    def output(self):
        path = ""
        if self.log_scale:
            path += "_log"
        if self.unblinded:
            path += "_data"
        return self.local_target("hists{}.pdf".format(path))

    def run(self):
        inp = self.input().load()
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
                    # ax = fig.add_subplot(number,1,i+1)
                    fig = plt.figure()
                    ax = fig.add_subplot()

                    # from IPython import embed;embed()

                    # ax.set_title("{0}: {1}".format(cat, var.name))
                    ax.autoscale(axis="x", tight=True)

                    if self.log_scale:
                        ax.set_yscale("log")
                        ax.set_ylim(0.0001, 1e6)

                    # if more datasets: FIXME
                    coffea.hist.plot1d(
                        inp[var.name][("tt", cat)].project(var.name),
                        ax=ax,
                        clear=True,
                    )
                    hep.set_style("CMS")
                    hep.cms.label(
                        llabel="Private Work",
                        lumi=np.round(self.config_inst.get_aux("lumi") / 1000.0, 2),
                        loc=0,
                        ax=ax,
                    )

                    # declare naming
                    leg = ax.legend(
                        title="{0}: {1}".format(cat, var.name),
                        # ncol=1,
                        loc="upper right",
                        # bbox_to_anchor=(1.04, 1),
                        # borderaxespad=0,
                    )
                    ax.set_xlabel(var.get_full_x_title())
                    ax.set_ylabel(var.get_full_y_title())

                    plt.tight_layout()
                    pdf.savefig(fig)
                    fig.clear()

            print("\n", " ---- Created {} pages ----".format(pdf.get_pagecount()), "\n")
