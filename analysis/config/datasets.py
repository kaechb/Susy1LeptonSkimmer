"""
define all needed datasets
"""

import order as od

# ttbar
# dataset_tt_sl = od.Dataset(
#    "tt_sl",
#    200,
#    campaign=campaign,
#    keys=[
#        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1/MINIAODSIM"
#    ],
# )


def setup_datasets(cfg, campaign):
    # build all variable histogram configuration to fill in coffea
    # each needs a name, a Matplotlib x title and a (#bins, start, end) binning

    cfg.add_dataset(
        "ttJets",
        1100,
        campaign=campaign,
        keys=["/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"],
    )
