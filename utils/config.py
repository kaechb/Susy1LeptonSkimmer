def variables():
    # build all variable histogram configuration to fill in coffea
    # each needs a name, a Matplotlib x title and a (#bins, start, end) binning

    dict = {

        "METPt": ( "METPt",
                   "MET$_{pt}$ [GeV]",
                   [150, 0, 750]
                   ),
        "W_mt": ("W_mt",
                 "W$_{mt}$ [GeV]",
                 [200,0,100]
                 )
    }

    return dict
