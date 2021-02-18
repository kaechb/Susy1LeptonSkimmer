from .variables import setup_variables
import scinum as sn
import order as od
import six
import copy

# base config
import config.Run2_pp_13TeV_2016 as run_2016

# create the analysis
analysis = od.Analysis("mj", 1)

config_2016 = cfg = analysis.add_config(run_2016.base_config.copy())

cfg.set_aux(
    "signal_process",
    "multijet",
)


setup_variables(cfg)
