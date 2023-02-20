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
        # keys=["/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"],
        keys=["/nfs/dust/cms/user/frengelk/Testing/TTJets_TuneCP5_RunIISummer19UL16NanoAODv2_1.root"],
    )

    cfg.add_dataset(
        "ttJets_2",
        1101,
        campaign=campaign,
        keys=["/nfs/dust/cms/user/frengelk/Testing/TTJets_TuneCP5_RunIISummer19UL16NanoAODAPVv2_1.root"],
    )


################
"""
    cfg.add_dataset(
        "ttJets_sl_fromt",
        1101,
        campaign=campaign,
        keys=["TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM/TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM_1_merged.root",
        "TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM/TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM_2_merged.root",
        "TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM/TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM_3_merged.root"
        "TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM/TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_RunIISummer16NanoAODv6-PUMoriond17_Nano25Oct2019_102X_mcRun2_asymptotic_v7-v1_NANOAODSIM_3_merged.root"



        ],
    ),
    cfg.add_dataset(
        "ttJets_sl_fromtbar",
        1102,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "ttJets_dilep",
        1103,
    ),
    cfg.add_dataset(
        "ttJets_HT600to800",
        1104,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "ttJets_HT800to1200",
        1105,
        campaign=campaign,
        keys=[],
    ),

    cfg.add_dataset(
        "ttJets_HT1200to2500",
        1106,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "ttJets_HT2500toInf",
        1107,
        campaign=campaign,
        keys=[],
    ),

    cfg.add_dataset(
        "QCD_HT100to200",
        1201,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "QCD_HT200to300",
        1202,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "QCD_HT300to500",
        1203,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "QCD_HT500to700",
        1204,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "QCD_HT700to1000",
        1205,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "QCD_HT1000to1500",
        1206,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "QCD_HT1500to2000",
        1207,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "QCD_HT2000toInf",
        1208,
        campaign=campaign,
        keys=[],
    ),

    cfg.add_dataset(
        "WJets_HT70to100",
        1301,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "WJets_HT100to200",
        1302,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "WJets_HT200to400",
        1303,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "WJets_HT400to600",
        1304,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "WJets_HT600to800",
        1305,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "WJets_HT800to1200",
        1306,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "WJets_HT1200to2500",
        1307,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "WJets_HT2500toInf",
        1308,
        campaign=campaign,
        keys=[],
    ),


    cfg.add_dataset(
        "DY_HT70to100",
        1401,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "DY_HT100to200",
        1402,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "DY_HT200to400",
        1403,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "DY_HT400to600",
        1404,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "DY_HT600to800",
        1405,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "DY_HT800to1200",
        1406,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "DY_HT1200to2500",
        1407,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "DY_HT2500toInf",
        1408,
        campaign=campaign,
        keys=[],
    ),


    cfg.add_dataset(
        "st_tW_top",
        1501,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "st_tW_antitop",
        1502,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "st_tW_antitop_no_fh",
        1503,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "st_tW_top_no_fh",
        1504,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "st_schannel_4f",
        505,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "st_tchannel_4f_incl",
        1506,
        campaign=campaign,
        keys=[],
    ),
    cfg.add_dataset(
        "st_tchannel_antitop_4f_incl",
        1507,
        campaign=campaign,
        keys=[],
    )
    """
