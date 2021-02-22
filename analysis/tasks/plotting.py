# coding: utf-8

import logging
import os
import law
import order as od
import law.contrib.coffea
from luigi import BoolParameter, Parameter
import coffea
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep

# other modules
from tasks.basetasks import *
from tasks.coffea import CoffeaProcessor


class PlotCoffeaHists(ConfigTask):
    def requires(self):
        return CoffeaProcessor.req(self, processor="Histogramer")

    def output(self):
        return self.local_target("hists.pdf")

    def run(self):
        inp = self.input().load()
        self.output().parent.touch()
        # create pdf object to save figures on separate pages
        with PdfPages(self.output().path) as pdf:
            # number = len(inp.keys())

            # plot each hist
            for var in self.config_inst.variables:

                hists = inp[var.name]
                categories = [h.name for h in hists.identifiers("category")]

                for cat in categories:
                    # ax = fig.add_subplot(number,1,i+1)
                    fig = plt.figure()
                    ax = fig.add_subplot()

                    # from IPython import embed;embed()

                    ax.set_title("{0}: {1}".format(cat, var.name))
                    # if more datasets: FIXME
                    coffea.hist.plot1d(
                        inp[var.name][("tt", cat)].project(var.name), ax=ax
                    )
                    hep.set_style("CMS")
                    hep.cms.label(
                        llabel="Private Work",
                        lumi=np.round(self.config_inst.get_aux("lumi") / 1000.0, 2),
                        loc=0,
                        ax=ax,
                    )
                    plt.tight_layout()

                    pdf.savefig(fig)
                    fig.clear()
