# coding: utf-8

import logging
import os
import law
import law.contrib.coffea
from luigi import BoolParameter, Parameter
from coffea.processor import ProcessorABC
import json
import time
import numpy as np
import matplotlib.pyplot as mpl

# other modules
from tasks.basetasks import AnalysisTask
from tasks.coffea import CoffeaProcessor


class PlotCoffeaHists(AnalysisTask):
    def requires(self):
        CoffeaProcessor(self, processor="Histogramer")

    def output(self):
        self.local_target("hists.pdf")

    def run(self):
        from IPython import embed

        embed()
