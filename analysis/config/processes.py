# coding: utf-8

"""
Physics processes.
If not stated otherwise, cross sections are given in pb.

declare processes by overall process
 --binned processes according to files

each process needs an unique number, (label) and a xsec (defined on the fileset)
"""


import order as od
import scinum as sn
from math import inf

from config.constants import *

"""
od.Process("data", 0, is_data=True, label=r"data", color=(0, 0, 0), processes=[
    od.Process("data_dl", 1, is_data=True, label=r"data", processes=[
        od.Process("data_ee", 3, is_data=True, label=r"data"),
        od.Process("data_emu", 4, is_data=True, label=r"data"),
        od.Process("data_mumu", 5, is_data=True, label=r"data"),
    ]),
    od.Process("data_sl", 2, is_data=True, label=r"data", processes=[
        od.Process("data_e", 6, is_data=True, label=r"data"),
        od.Process("data_mu", 7, is_data=True, label=r"data"),
    ]),
])
"""


def setup_processes(cfg):
    # build processes
    cfg.add_process(
        "ttJets",
        100,
        label=r"$t\Bar{t}$ Jets",
        label_short="ttJ",
        color=(1, 90, 184),
        xsecs={
            13: sn.Number(1234.0 / 56, ("rel", 0.07)),
        },
    )
