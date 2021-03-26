def setup_variables(cfg):
    """
    build all variable histogram configuration to fill in coffea
    each needs a name, a Matplotlib x title and a (#bins, start, end) binning
    template
    cfg.add_variable(
        name="",
        expression="",
        binning=(, , ),
        unit="",
        x_title=r"",
    )
    """

    cfg.add_variable(
        name="METPt",
        expression="METPt",
        binning=(50, 0.0, 1000),
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

    cfg.add_variable(
        name="LT",
        expression="LT",
        binning=(50, 0.0, 1000),
        unit="GeV",
        x_title=r"LT",
    )

    cfg.add_variable(
        name="HT",
        expression="HT",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"HT",
    )

    cfg.add_variable(
        name="jet_mass_1",
        expression="jet_mass_1",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"m_{Jet}^{1}",
    )


    cfg.add_variable(
        name="n_jets",
        expression="n_jets",
        binning=(20,0 ,20 ),
        #unit="",
        x_title=r"Number of Jets",
    )

    cfg.add_variable(
        name="Dphi",
        expression="Dphi",
        binning=(64,-3.2 ,3.2 ),
        #unit="",
        x_title=r"\Delta \Phi",
    )

    cfg.add_variable(
        name="lead_lep_pt",
        expression="lead_lep_pt",
        binning=(100,0 ,1000 ),
        unit="",
        x_title=r"",
    )
