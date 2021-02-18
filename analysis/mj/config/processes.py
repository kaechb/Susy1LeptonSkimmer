"""
Common process definitions.
"""


__all__ = [
    "process_singleTop",
    "process_VJets",
    "process_WJets",
    "process_ZJets",
    "process_VVJets",
    "process_WWJets",
    "process_WZJets",
    "process_ZZJets",
]


import order as od
import scinum as sn

#
# define processes
# (cross sections are made up as all datasets in this example correspond to 50/fb)
#

process_singleTop = od.Process(
    "ttJets",
    1000,
    label=r"$t/\bar{t}$ Jets",
    label_short="ttJ",
    color=(1, 90, 184),
    xsecs={
        13: sn.Number(1234.0 / 56, ("rel", 0.07)),
    },
)
