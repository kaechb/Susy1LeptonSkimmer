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
    """
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
    cfg.add_process(
        "ttJets_2",
        101,
        label=r"$t\Bar{t}$ Jets",
        label_short="ttJ",
        color=(90, 120, 150),
        xsecs={
            13: sn.Number(2468.0 / 56, ("rel", 0.07)),
        },
    )
    """

    ############

    cfg.add_process(
        "ttJets",
        100,
        label=r"tt+Jets",
        label_short="ttJ",
        color=(50, 100, 150),
        processes=[
            od.Process(
                "ttJets_sl_fromt",
                101,
                label=r"ttJets sl t",
                xsecs={
                    13: sn.Number(182.18),
                },
            ),
            od.Process(
                "ttJets_sl_fromtbar",
                102,
                label=r"ttJets sl tbar",
                xsecs={
                    13: sn.Number(182.18),
                },
            ),
            od.Process(
                "ttJets_dilep",
                103,
                label=r"ttJets dl",
                xsecs={
                    13: sn.Number(87.315),
                },
            ),
            od.Process(
                "ttJets_HT600to800",
                104,
                label=r"ttJets HT 600-800",
                xsecs={
                    13: sn.Number(2.76),
                },
            ),
            od.Process(
                "ttJets_HT800to1200",
                105,
                label=r"ttJets HT 800-1200",
                xsecs={
                    13: sn.Number(1.116),
                },
            ),
            od.Process(
                "ttJets_HT1200to2500",
                106,
                label=r"ttJets HT 1200-2500",
                xsecs={
                    13: sn.Number(0.198),
                },
            ),
            od.Process(
                "ttJets_HT2500toInf",
                107,
                label=r"ttJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.002),
                },
            ),
        ],
    )

    cfg.add_process(
        "QCD",
        200,
        label=r"QCD Multijet",
        label_short="QCD",
        color=(150, 100, 50),
        processes=[
            od.Process(
                "QCD_HT100to200",
                201,
                label=r"QCD HT 100-200",
                xsecs={
                    13: sn.Number(27990000),
                },
            ),
            od.Process(
                "QCD_HT200to300",
                202,
                label=r"QCD HT 200-300",
                xsecs={
                    13: sn.Number(1712000),
                },
            ),
            od.Process(
                "QCD_HT300to500",
                203,
                label=r"QCD HT 300-500",
                xsecs={
                    13: sn.Number(347700),
                },
            ),
            od.Process(
                "QCD_HT500to700",
                204,
                label=r"QCD HT 500-700",
                xsecs={
                    13: sn.Number(32100),
                },
            ),
            od.Process(
                "QCD_HT700to1000",
                205,
                label=r"QCD HT 700-1000",
                xsecs={13: sn.Number(6831)},
            ),
            od.Process(
                "QCD_HT1000to1500",
                206,
                label=r"QCD HT 1000-1500",
                xsecs={
                    13: sn.Number(1207),
                },
            ),
            od.Process(
                "QCD_HT1500to2000",
                207,
                label=r"QCD HT 1500-2000",
                xsecs={13: sn.Number(119.9)},
            ),
            od.Process(
                "QCD_HT2000toInf",
                208,
                label=r"QCD HT 2000-Inf",
                xsecs={
                    13: sn.Number(25.24),
                },
            ),
        ],
    )

    cfg.add_process(
        "WJets",
        300,
        label=r"W+Jets \rightarrow l \nu ",
        label_short="W+JEts",
        color=(200, 200, 200),
        processes=[
            od.Process(
                "WJets_HT70to100",
                301,
                label=r"WJets HT 70-100",
                xsecs={
                    13: sn.Number(1353),
                },
            ),
            od.Process(
                "WJets_HT100to200",
                302,
                label=r"WJets HT 100-200",
                xsecs={
                    13: sn.Number(1627.45),
                },
            ),
            od.Process(
                "WJets_HT200to400",
                303,
                label=r"WJets HT 200-400",
                xsecs={
                    13: sn.Number(435.237),
                },
            ),
            od.Process(
                "WJets_HT400to600",
                304,
                label=r"WJets HT 400-600",
                xsecs={
                    13: sn.Number(59.181),
                },
            ),
            od.Process(
                "WJets_HT600to800",
                305,
                label=r"WJets HT 600-800",
                xsecs={13: sn.Number(14.581)},
            ),
            od.Process(
                "WJets_HT800to1200",
                306,
                label=r"WJets HT 800-1200",
                xsecs={
                    13: sn.Number(6.656),
                },
            ),
            od.Process(
                "WJets_HT1200to2500",
                307,
                label=r"WJets HT 1200-2500",
                xsecs={13: sn.Number(1.608)},
            ),
            od.Process(
                "WJets_HT2500toInf",
                308,
                label=r"WJets HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.039),
                },
            ),
        ],
    )

    cfg.add_process(
        "DY",
        400,
        label=r"DY \rightarrow l l ",
        label_short="DY",
        color=(100, 100, 100),
        processes=[
            od.Process(
                "DY_HT70to100",
                401,
                label=r"DY HT 70-100",
                xsecs={
                    13: sn.Number(208.977),
                },
            ),
            od.Process(
                "DY_HT100to200",
                402,
                label=r"DY HT 100-200",
                xsecs={
                    13: sn.Number(181.305),
                },
            ),
            od.Process(
                "DY_HT200to400",
                403,
                label=r"DY HT 200-400",
                xsecs={
                    13: sn.Number(50.148),
                },
            ),
            od.Process(
                "DY_HT400to600",
                404,
                label=r"DY HT 400-600",
                xsecs={
                    13: sn.Number(6.984),
                },
            ),
            od.Process(
                "DY_HT600to800",
                405,
                label=r"DY HT 600-800",
                xsecs={13: sn.Number(1.681)},
            ),
            od.Process(
                "DY_HT800to1200",
                406,
                label=r"DY HT 800-1200",
                xsecs={
                    13: sn.Number(0.775),
                },
            ),
            od.Process(
                "DY_HT1200to2500",
                407,
                label=r"DY HT 1200-2500",
                xsecs={13: sn.Number(0.186)},
            ),
            od.Process(
                "DY_HT2500toInf",
                408,
                label=r"DY HT 2500-Inf",
                xsecs={
                    13: sn.Number(0.004),
                },
            ),
        ],
    )

    cfg.add_process(
        "st",
        500,
        label=r"Single t SM",
        label_short="st",
        color=(150, 150, 150),
        processes=[
            od.Process(
                "st_tW_top",
                501,
                label=r"st tW top",
                xsecs={
                    13: sn.Number(35.85),
                },
            ),
            od.Process(
                "st_tW_antitop",
                502,
                label=r"st tW antitop",
                xsecs={
                    13: sn.Number(35.85),
                },
            ),
            od.Process(
                "st_tW_antitop_no_fh",
                503,
                label=r"s antitop no fh",
                xsecs={
                    13: sn.Number(16.295),
                },
            ),
            od.Process(
                "st_tW_top_no_fh",
                504,
                label=r"s top no fh",
                xsecs={
                    13: sn.Number(16.295),
                },
            ),
            od.Process(
                "st_schannel_4f",
                505,
                label=r"st s 4f",
                xsecs={13: sn.Number(3.360)},
            ),
            od.Process(
                "st_tchannel_4f_incl",
                506,
                label=r"st t_ch incl",
                xsecs={
                    13: sn.Number(136.02),
                },
            ),
            od.Process(
                "st_tchannel_antitop_4f_incl",
                507,
                label=r"santit t_ch incl",
                xsecs={13: sn.Number(80.97)},
            ),
        ],
    )


"""


        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )

        cfg.add_process(
        ,
        ,
        label=r"",
        label_short="",
        color=(,,),
        xsecs={
            13: sn.Number()),
        },
    )
"""
