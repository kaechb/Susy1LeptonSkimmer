# coding: utf-8


def setup_categories(cfg):

    N1b_SR = cfg.add_category(
        "N1b_SR",
        label="1i btagged jet, signal region",
        label_short="1i btag, SR",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
    N1b_CR = cfg.add_category(
        "N1b_CR",
        label="1i btagged jet, control region",
        label_short="1i btag, CR",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
