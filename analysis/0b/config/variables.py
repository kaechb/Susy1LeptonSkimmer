########################################################################################
# Setup Signal bins for the analysis                                                   #
########################################################################################
def setup_variables(cfg):
    """
    build all variable histogram configuration to fill in coffea
    each needs a name, a Matplotlib x title and a (#bins, start, end) binning
    template
    cfg.add_variable(name="", expression="", binning=(, , ), unit="", x_title=r"")
    """
    cfg.add_variable(name="metPt", expression="metPt", binning=(50, 0.0, 1000), unit="GeV", x_title=r"$p_{T}^{miss}$")
    cfg.add_variable(name="WBosonMt", expression="WBosonMt", binning=(50, 0.0, 1000.0), unit="GeV", x_title=r"$m_{t}^{W}$")
    cfg.add_variable(name="LT", expression="LT", binning=(50, 0.0, 2000), unit="GeV", x_title="LT")
    cfg.add_variable(name="HT", expression="HT", binning=(50, 0.0, 4000.0), unit="GeV", x_title="HT")
    cfg.add_variable(name="nJets", expression="nJets", binning=(20, 0, 20))
    cfg.add_variable(name="nWFatJets", expression="nWFatJets", binning=(20, 0, 20))
    cfg.add_variable(name="ntFatJets", expression="ntFatJets", binning=(20, 0, 20))
    # lepton stuff ###############
    cfg.add_variable(name="nMuon", expression="nMuon", binning=(10, 0, 10))
    cfg.add_variable(name="nElectron", expression="nElectron", binning=(10, 0, 10))
    cfg.add_variable(name="leadMuonPt", expression="leadMuonPt", binning=(100, 0, 1000), unit="GeV", x_title=r"$p_{T}^{lep1}$")
    cfg.add_variable(name="leadMuonEta", expression="leadMuonEta", binning=(20, 0, 5), unit="GeV", x_title=r"$\eta^{lep1}$")
    cfg.add_variable(name="leadMuonPhi", expression="leadMuonPhi", binning=(63, -3.15, 3.15), unit="GeV", x_title=r"$\Phi^{lep1}$")
    cfg.add_variable(name="leadElectronPt", expression="leadElectronPt", binning=(100, 0, 1000), unit="GeV", x_title=r"$p_{T}^{lep1}$")
    cfg.add_variable(name="leadElectronEta", expression="leadElectronEta", binning=(20, 0, 5), unit="GeV", x_title=r"$\eta^{lep1}$")
    cfg.add_variable(name="leadElectronPhi", expression="leadElectronPhi", binning=(63, -3.15, 3.15), unit="GeV", x_title=r"$\Phi^{lep1}$")
    # jet stuff ##################
    cfg.add_variable(name="jetMass_1", expression="jetMass_1", binning=(50, 0.0, 1000.0), unit="GeV", x_title=r"$m_{Jet}^{1}$")
    cfg.add_variable(name="jetPt_1", expression="jetPt_1", binning=(100, 0.0, 1000.0), unit="GeV", x_title=r"$p_{T}^{Jet1}$")
    cfg.add_variable(name="jetEta_1", expression="jetEta_1", binning=(20, 0.0, 5.0), unit="GeV", x_title=r"$\eta_{Jet}^{1}$")
    cfg.add_variable(name="jetPhi_1", expression="jetPhi_1", binning=(63, -3.15, 3.15), unit="GeV", x_title=r"$\Phi_{Jet}^{1}$")
    cfg.add_variable(name="jetMass_2", expression="jetMass_1", binning=(50, 0.0, 1000.0), unit="GeV", x_title=r"$m_{Jet}^{1}$")
    cfg.add_variable(name="jetPt_2", expression="jetPt_2", binning=(100, 0.0, 1000.0), unit="GeV", x_title=r"$p_{T}^{Jet2}$")
    cfg.add_variable(name="jetEta_2", expression="jetEta_1", binning=(20, 0.0, 5.0), unit="GeV", x_title=r"$\eta_{Jet}^{1}$")
    cfg.add_variable(name="jetPhi_2", expression="jetPhi_1", binning=(63, -3.15, 3.15), unit="GeV", x_title=r"$\Phi_{Jet}^{1}$")
    #########################
    cfg.add_variable(name="dPhi", expression="dPhi", binning=(64, -3.2, 3.2), x_title=r"$ \Delta \Phi $")
