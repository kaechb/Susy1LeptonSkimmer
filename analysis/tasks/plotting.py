# coding: utf-8

import os
import law
import order as od
import luigi
import coffea
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep
from tqdm.auto import tqdm
import operator

# other modules
from tasks.basetasks import *
from tasks.coffea import CoffeaProcessor


class PlotCoffeaHists(ConfigTask):

    log_scale = luigi.BoolParameter()
    unblinded = luigi.BoolParameter()
    scale_signal = luigi.IntParameter(default=1)
    debug = luigi.BoolParameter()

    def requires(self):
        if self.debug:
            return CoffeaProcessor.req(
                self, processor="Histogramer", workflow="local", debug=True
            )

        else:
            return CoffeaProcessor.req(self, processor="Histogramer")

    def output(self):
        path = ""
        if self.log_scale:
            path += "_log"
        if self.unblinded:
            path += "_data"
        if self.debug:
            path += "_debug"
        return self.local_target("hists{}.pdf".format(path))

    def run(self):
        inp = self.input()["collection"][0].load()
        self.output().parent.touch()

        # setting style
        # plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "sans-serif",
        # "font.sans-serif": ["Helvetica"]})

        # declare sytling options
        error_opts = {
            "label": "Stat. Unc.",
            "hatch": "///",
            "facecolor": "none",
            "edgecolor": (0, 0, 0, 0.5),
            "linewidth": 0,
        }
        data_err_opts = {
            "linestyle": "none",
            "marker": ".",
            "markersize": 10.0,
            "color": "k",
            "elinewidth": 1,
        }
        line_opts = {
            "linestyle": "-",
            "color": "r",
        }

        # create pdf object to save figures on separate pages
        with PdfPages(self.output().path) as pdf:

            # from IPython import embed;embed()
            # plot each hist
            for var in tqdm(self.config_inst.variables, unit="variable"):
                # print(var.get_full_x_title())
                hists = inp[var.name]
                categories = [h.name for h in hists.identifiers("category")]

                for cat in categories:

                    # lists for collecting and sorting hists
                    bg_hists = []
                    hists_attr = []
                    data_hists = []

                    # enlargen figure
                    # plt.figure(figsize=(8, 6), dpi=80)

                    if self.unblinded:
                        # build canvas for data plotting
                        fig, (ax, rax) = plt.subplots(
                            2,
                            1,
                            figsize=(18, 10),
                            sharex=True,
                            gridspec_kw=dict(
                                height_ratios=(3, 1),
                                hspace=0,
                            ),
                        )

                    else:
                        fig, ax = plt.subplots(figsize=(18, 10))

                    for proc in self.config_inst.processes:

                        if "data" in proc.name and self.unblinded:
                            child_hists = hists[
                                [p[0].name for p in proc.walk_processes()], cat
                            ]
                            mapping = {
                                proc.label: [
                                    p[0].name
                                    for p in proc.walk_processes()
                                    if not "B" in p[0].name
                                ]
                            }
                            grouped = child_hists.group(
                                "dataset",
                                coffea.hist.Cat("process", proc.label_short),
                                mapping,
                            ).integrate("category")

                            data_hists.append(grouped)

                        if not self.debug and not "data" in proc.name:
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
                                coffea.hist.Cat("process", proc.label_short),
                                mapping,
                            ).integrate("category")

                            bg_hists.append(grouped)
                            hists_attr.append(proc.color)

                        if self.debug:
                            # from IPython import embed;embed()
                            dat = hists.identifiers("dataset")
                            bg = hists[(str(dat[0]), cat)].integrate("category")
                            # for dat in hists.identifiers("dataset"):

                    if not self.debug:
                        bg = bg_hists[0]
                        for i in range(1, len(bg_hists)):
                            bg.add(bg_hists[i])

                    # from IPython import embed;embed()
                    # order the processes by magnitude of integral
                    order_lut = bg.integrate(var.name).values()
                    bg_order = sorted(order_lut.items(), key=operator.itemgetter(1))
                    order = [name[0][0] for name in bg_order]

                    # plot bg
                    coffea.hist.plot1d(
                        # inp[var.name].integrate("category"),
                        bg,
                        ax=ax,
                        stack=True,
                        overflow="none",
                        fill_opts=dict(color=[col for col in hists_attr]),
                        order=order,
                        # fill_opts=hists_attr,   #dict(color=proc.color),
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
                    if self.unblinded:
                        # normally, we have only two kinds of data, if more, adapt
                        dat = data_hists[0].add(data_hists[1])
                        data = dat.group(
                            "process",
                            coffea.hist.Cat("process", "data"),
                            {"data": ["data electron", "data muon"]},
                        )
                        coffea.hist.plot1d(
                            data,  # .project( "process", varName),
                            # overlay="process",
                            error_opts=data_err_opts,
                            ax=ax,
                            # overflow="none",
                            # binwnorm=True,
                            clear=False,
                        )
                        coffea.hist.plotratio(
                            data.sum("process"),
                            bg.sum("process"),
                            ax=rax,
                            error_opts=data_err_opts,
                            denom_fill_opts={},
                            guide_opts={},
                            unc="num",
                            clear=False,
                        )
                        rax.set_ylabel("Ratio")
                        rax.set_ylim(0, 2)

                    # declare naming
                    leg = ax.legend(
                        title="{0}: {1}".format(cat, var.x_title),
                        ncol=1,
                        loc="upper left",
                        bbox_to_anchor=(1, 1),
                        borderaxespad=0,
                    )

                    if self.log_scale:
                        ax.set_yscale("log")
                        ax.set_ylim(0.0001, 1e12)
                        # FIXME be careful with logarithmic ratios
                        # if self.unblinded:
                        #    rax.set_yscale("log")
                        #    rax.set_ylim(0.0001, 1e0)

                    ax.set_xlabel(var.get_full_x_title())
                    ax.set_ylabel(var.get_full_y_title())
                    ax.autoscale(axis="x", tight=True)

                    if self.unblinded:
                        rax.set_xlabel(var.get_full_x_title())
                        rax.set_ylabel("Ratio")
                        rax.autoscale(axis="x", tight=True)

                    plt.tight_layout()
                    pdf.savefig(fig)

                    ax.cla()
                    if self.unblinded:
                        rax.cla()
                    plt.close(fig)

            print("\n", " ---- Created {} pages ----".format(pdf.get_pagecount()), "\n")


class ArrayPlotting(ConfigTask):
    def requires(self):
        return CoffeaProcessor.req(
            self, processor="ArrayExporter", debug=True, workflow="local"
        )

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
