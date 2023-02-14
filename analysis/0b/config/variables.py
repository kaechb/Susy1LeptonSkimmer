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
        binning=(50, 0.0, 2000),
        unit="GeV",
        x_title="LT",
    )

    cfg.add_variable(
        name="HT",
        expression="HT",
        binning=(50, 0.0, 4000.0),
        unit="GeV",
        x_title="HT",
    )

    cfg.add_variable(
        name="n_jets",
        expression="n_jets",
        binning=(20, 0, 20),
        # unit="",
        x_title="Number of Jets",
    )

    # lepton stuff ###############
    cfg.add_variable(
        name="n_muon",
        expression="n_muon",
        binning=(10, 0, 10),
        # unit="",
        x_title="Number of Muons",
    )

    cfg.add_variable(
        name="n_electron",
        expression="n_electron",
        binning=(10, 0, 10),
        # unit="",
        x_title="Number of Electrons",
    )

    cfg.add_variable(
        name="lead_lep_pt",
        expression="lead_lep_pt",
        binning=(100, 0, 1000),
        unit="GeV",
        x_title=r"$p_{T}^{lep1}$",
    )

    cfg.add_variable(
        name="lead_lep_eta",
        expression="lead_lep_eta",
        binning=(20, 0, 5),
        unit="GeV",
        x_title=r"$\eta^{lep1}$",
    )

    cfg.add_variable(
        name="lead_lep_phi",
        expression="lead_lep_phi",
        binning=(63, -3.15, 3.15),
        unit="GeV",
        x_title=r"$\Phi^{lep1}$",
    )

    # jet stuff ##################
    cfg.add_variable(
        name="jet_mass_1",
        expression="jet_mass_1",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$m_{Jet}^{1}$",
    )

    cfg.add_variable(
        name="jet_pt_1",
        expression="jet_pt_1",
        binning=(100, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$p_{T}^{Jet1}$",
    )

    cfg.add_variable(
        name="jet_pt_2",
        expression="jet_pt_2",
        binning=(100, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$p_{T}^{Jet2}$",
    )

    cfg.add_variable(
        name="jet_eta_1",
        expression="jet_eta_1",
        binning=(20, 0.0, 5.0),
        unit="GeV",
        x_title=r"$\eta_{Jet}^{1}$",
    )

    cfg.add_variable(
        name="jet_phi_1",
        expression="jet_phi_1",
        binning=(63, -3.15, 3.15),
        unit="GeV",
        x_title=r"$\Phi_{Jet}^{1}$",
    )
    #########################

    cfg.add_variable(
        name="Dphi",
        expression="Dphi",
        binning=(64, -3.2, 3.2),
        # unit="",
        x_title=r"$ \Delta \Phi $",
    )
