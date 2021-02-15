def setup_variables(cfg):
    # build all variable histogram configuration to fill in coffea
    # each needs a name, a Matplotlib x title and a (#bins, start, end) binning

    cfg.add_variable(
    name="MET",
    expression="METPt",
    binning=(30, 0.0, 750),
    unit="GeV",
    x_title=r"$p_{T}^{miss}$",
    )


    cfg.add_variable(
    name="W_mt",
    expression="W_mt",
    binning=(50, 0.0, 1000.0),
    unit="GeV",
    x_title=r"$m_{t}^{W}$",
    )
