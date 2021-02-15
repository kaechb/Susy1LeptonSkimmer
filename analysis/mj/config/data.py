# coding: utf-8

"""
Common campaign and dataset definitions for 2011 Open Data @ 7 TeV.
"""


__all__ = [
    "campaign_opendata_2011", "dataset_singleTop", "dataset_WJets", "dataset_ZJets",
    "dataset_WWJets", "dataset_WZJets", "dataset_ZZJets",
]


import order as od

#import analysis.config.processes as pro
# define datasets
# (n_files is set artificially to have <= 5k events per file)
#

dataset_singleTop = cp.add_dataset("ttJets", 210,
    processes=[procs.process_ttJets],
    keys=["/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"],
    n_files=1,
    #n_events=5684,
)

