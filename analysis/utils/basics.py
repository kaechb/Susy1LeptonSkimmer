"""
Some helper functions.
"""


__all__ = [
    "load_ROOT",
    "parse_leaf_list",
    "get_tree_names",
    "get_trees",
    "copy_trees",
]


import os
import re
import array
import collections
import subprocess

import law


_ROOT = None


def load_ROOT():
    """
    Loads the ROOT package once, sets common package configurations, then caches and returns it.
    """
    global _ROOT

    if _ROOT is None:
        import ROOT

        ROOT.PyConfig.IgnoreCommandLineOptions = True
        ROOT.gROOT.SetBatch()
        _ROOT = ROOT

    return _ROOT


def calc_checksum(*paths, **kwargs):
    exclude = law.util.make_list(kwargs.get("exclude", ["*.pyc", "*.git*"]))
    exclude = " ".join("! -path '{}'".format(p) for p in exclude)

    sums = []
    for path in paths:
        path = os.path.expandvars(os.path.expanduser(path))
        if os.path.isfile(path):
            cmd = 'sha1sum "{}"'.format(path)
        elif os.path.isdir(path):
            cmd = (
                'files="$( find "{}" -type f {} -print | sort -z )"; '
                "(for f in $files; do sha1sum $f; done) | sha1sum".format(path, exclude)
            )
        else:
            raise IOError("file or directory '{}' does not exist".format(path))

        code, out, _ = law.util.interruptable_popen(
            cmd, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
        )
        if code != 0:
            raise Exception("checksum calculation failed")

        sums.append(out.strip().split(" ")[0])

    if len(sums) == 1:
        return sums[0]
    else:
        cmd = 'echo "{}" | sha1sum'.format(",".join(sums))
        code, out, _ = law.util.interruptable_popen(
            cmd, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
        )
        if code != 0:
            raise Exception("checksum combination failed")

        return out.strip().split(" ")[0]


root_array_types = {"D": "d", "F": "f", "I": "i"}


def parse_leaf_list(leaf_list, default_type="D"):
    """
    Parses a string *leaf_list* (e.g. ``"leafA/F:leafB/D:leafC"``) and returns a list of tuples
    containing information in the leaves: ``(name, root_type, python_type)``. Supported ROOT types
    are ``F``, ``D`` and ``I``.
    """
    data = []
    for leaf in leaf_list.split(":"):
        parts = leaf.rsplit("/", 1)
        if len(parts) == 1:
            parts.append(default_type)
        name, root_type = parts
        if root_type not in root_array_types:
            raise Exception("unknown root type: " + root_type)
        data.append((name, root_type, root_array_types[root_type]))
    return data


def get_tree_names(tfile, name_pattern="*"):
    """
    Returns the names of all trees found in a *tfile* that pass *name_pattern*.
    """
    ROOT = load_ROOT()

    names = []
    for tkey in tfile.GetListOfKeys():
        name = tkey.GetName()

        if not law.util.multi_match(name, name_pattern):
            continue

        tobj = tfile.Get(name)
        if not isinstance(tobj, ROOT.TTree):
            continue

        names.append(name)

    return names


def get_trees(tfile, name_pattern="*"):
    """
    Returns all trees found in a *tfile* that pass *name_pattern*.

    .. code-block:: python

        tfile = ROOT.TFile("/tmp/file.root")
        get_trees(tfile)
        # -> [<ROOT.TTree object ("myTreeA") at 0x2cb6400>,
        #     <ROOT.TTree object ("myTreeB") at 0x2df6420>,
        #     <ROOT.TTree object ("fooTree") at 0x2aa6480>,
        #     ... ]

        get_trees(tfile, "myTree*")
        # -> [<ROOT.TTree object ("myTreeA") at 0x2cb6400>,
        #     <ROOT.TTree object ("myTreeB") at 0x2df6420>]
    """
    names = get_tree_names(tfile, name_pattern=name_pattern)
    return [tfile.Get(name) for name in names]


def copy_trees(src, dst, name_pattern="*", force=False):
    """
    Copies all trees from a *src* file that match *name_pattern* into an other file *dst*. When the
    target file exists and *force* is *False*, an *IOError* is raised.
    """
    ROOT = load_ROOT()

    src = os.path.expandvars(os.path.expanduser(src))
    dst = os.path.expandvars(os.path.expanduser(dst))

    if os.path.exists(dst):
        if not force:
            raise IOError("destination file '{}' exists, force is False".format(dst))
        else:
            os.remove(dst)

    # simple cp when all trees should be copied
    tfile_src = ROOT.TFile.Open(src, "READ")
    all_names = get_tree_names(tfile_src)
    names = get_tree_names(tfile_src, name_pattern=name_pattern)
    if len(names) == len(all_names):
        tfile_src.Close()
        ROOT.TFile.Cp(src, dst, ROOT.kFALSE)
        return dst

    # create the dst directory
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # open the dst file
    tfile_dst = ROOT.TFile.Open(dst, "RECREATE")
    tfile_dst.cd()

    # copy all input trees
    for name in names:
        tree = tfile_src.Get(name)
        tfile_dst.cd()
        copy = tree.CloneTree(-1, "fast")
        copy.SetName(tree.GetName())
        copy.Write()

    tfile_dst.Close()
    tfile_src.Close()

    return dst


def SetTDRStyle():
    """Sets the PubComm recommended style
    Just a copy of <http://ghm.web.cern.ch/ghm/plots/MacroExample/tdrstyle.C>
    """
    ROOT = load_ROOT()
    # For the canvas:
    ROOT.gStyle.SetOptStat(0)

    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
    ROOT.gStyle.SetCanvasDefH(600)
    ROOT.gStyle.SetCanvasDefW(600)
    ROOT.gStyle.SetCanvasDefX(0)
    ROOT.gStyle.SetCanvasDefY(0)

    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.13)
    ROOT.gStyle.SetPadLeftMargin(0.16)
    ROOT.gStyle.SetPadRightMargin(0.05)

    ROOT.gStyle.SetHistLineColor(1)
    ROOT.gStyle.SetHistLineStyle(0)
    ROOT.gStyle.SetHistLineWidth(1)
    ROOT.gStyle.SetEndErrorSize(2)
    ROOT.gStyle.SetMarkerStyle(20)

    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTitleFont(42)
    ROOT.gStyle.SetTitleColor(1)
    ROOT.gStyle.SetTitleTextColor(1)
    ROOT.gStyle.SetTitleFillColor(10)
    ROOT.gStyle.SetTitleFontSize(0.05)

    ROOT.gStyle.SetTitleColor(1, "XYZ")
    ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetTitleXOffset(1.00)
    ROOT.gStyle.SetTitleYOffset(1.60)

    ROOT.gStyle.SetLabelColor(1, "XYZ")
    ROOT.gStyle.SetLabelFont(42, "XYZ")
    ROOT.gStyle.SetLabelOffset(0.007, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")

    ROOT.gStyle.SetAxisColor(1, "XYZ")
    ROOT.gStyle.SetStripDecimals(True)
    ROOT.gStyle.SetTickLength(0.03, "XYZ")
    ROOT.gStyle.SetNdivisions(510, "XYZ")
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)

    ROOT.gStyle.SetPaperSize(20.0, 20.0)
    ROOT.gStyle.SetHatchesLineWidth(5)
    ROOT.gStyle.SetHatchesSpacing(0.05)

    ROOT.TGaxis.SetExponentOffset(-0.08, 0.01, "Y")
