# coding: utf-8


def setup_categories(cfg):

    N0b_SR = cfg.add_category(
        "N0b_SR",
        label="0 btagged jets, signal region",
        label_short="0 btag, SR",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
    N1b_SR = cfg.add_category(
        "N1b_SR",
        label="1i btagged jet, signal region",
        label_short="1i btag, SR",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )

    N0b_CR = cfg.add_category(
        "N0b_CR",
        label="0 btagged jets, control region",
        label_short="0 btag, CR",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
    N1b_CR = cfg.add_category(
        "N1b_CR",
        label="1i btagged jet, control region",
        label_short="1i btag, CR",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )


"""
    NW0 = cfg.add_category(
        "NW0",
        label="0 Wtagged jets",
        label_short="0 Wtag",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
    NW1i = cfg.add_category(
        "NW1i",
        label="1i Wtagged jet",
        label_short="1i Wtag",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
"""
