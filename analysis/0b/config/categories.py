# coding: utf-8


def setup_categories(cfg):

    N0b = cfg.add_category(
        "N0b",
        label="0 btagged jets",
        label_short="0 btag",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )
    N1b = cfg.add_category(
        "N1b",
        label="1i btagged jet",
        label_short="1i btag",
        aux={"fit": True, "fit_variable": "jet1_pt"},
    )

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
