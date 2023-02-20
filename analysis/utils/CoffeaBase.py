import coffea

# from coffea.processor import ProcessorABC
# import law
import numpy as np
import uproot as up
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import hist, processor
from coffea.hist.hist_tools import DenseAxis, Hist

from coffea.processor.accumulator import (
    dict_accumulator,
    defaultdict_accumulator,
    column_accumulator,
    set_accumulator,
)
from coffea.processor.executor import WorkQueueExecutor

# register our candidate behaviors
# from coffea.nanoevents.methods import candidate
# ak.behavior.update(candidate.behavior)


class BaseProcessor(processor.ProcessorABC):
    individal_weights = False

    # jes_shifts = False
    # dataset_shifts = False

    def __init__(self, task):
        # self.publish_message = task.publish_message if task.debug else None
        self.config = task.config_inst
        # self.corrections = task.load_corrections()
        self.datasetAxis = hist.Cat("dataset", "Primary dataset")
        # self.dataset_shift_axis = hist.Cat("dataset_shift", "Dataset shift")
        self.categoryAxis = hist.Cat("category", "Category selection")
        # self.syst_axis = hist.Cat("systematic", "Shift of systematic uncertainty")
        self._accumulator = dict_accumulator(
            n_events=defaultdict_accumulator(int),
            sum_gen_weights=defaultdict_accumulator(float),
            object_cutflow=defaultdict_accumulator(int),
            cutflow=hist.Hist("Counts", self.datasetAxis, self.categoryAxis, self.categoryAxis, hist.Bin("cutflow", "Cut index", 10, 0, 10)),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def GetDataset(self, events):
        return events.metadata["dataset"]

    # def GetDatasetShift(self, events):
    #     return events.metadata["dataset"][1]

class BaseSelection:
    hl = ("MetPt","W_mt")

    # dtype = np.float32
    debug_dataset = (
        # "QCD_HT100to200"  # "TTTT""QCD_HT200to300"  # "TTTo2L2Nu"  # "WWW_4F"  # "data_B_ee"
    )

    def ObjGetSelectedVariables(self, X, n, extra=()):
        #TODO
        pass

    def GetSelectedVariables(self, X):
        return dict(hl=np.stack([ak.to_numpy(X[var]).astype(np.float32) for var in self.config.variables.names()],axis=-1))

    def AddToSelection(self, selection, name, array):
        return selection.add(name, ak.to_numpy(array, allow_missing=True))
    
    def GetBaseVariables(self, events):
        ntFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagTvsQCD), value=0)
        nWFatJets = ak.fill_none(ak.firsts(events.FatJetDeepTagWvsQCD), value=0)
        jetMass_1 = ak.fill_none(ak.firsts(events.JetMass[:, 0:1]), value=0)
        jetPt_1 = ak.fill_none(ak.firsts(events.JetPt[:, 0:1]), value=0)
        JetEta_1 = ak.fill_none(ak.firsts(events.JetEta[:, 0:1]), value=0)
        JetPhi_1 = ak.fill_none(ak.firsts(events.JetPhi[:, 0:1]), value=0)
        jetMass_2 = ak.fill_none(ak.firsts(events.JetMass[:, 1:2]), value=0)
        JetPt_2 = ak.fill_none(ak.firsts(events.JetPt[:, 1:2]), value=0)
        JetEta_2 = ak.fill_none(ak.firsts(events.JetEta[:, 1:2]), value=0)
        JetPhi_2 = ak.fill_none(ak.firsts(events.JetPhi[:, 1:2]), value=0)
        nJets = events.nJet
        return locals()

    def GetGenVariables(self, events):
        genMetPt = events.GenMetPt
        genMetPhi = events.GenMetPhi
        genJetPt_1 = events.GenJetPt_1
        genJetPhi_1 = events.GenJetPhi_1
        genJetEta_1 = events.GenJetEta_1
        genJetMass_1 = events.GenJetMass_1
        genJetPt_2 = events.GenJetPt_2
        genJetPhi_2 = events.GenJetPhi_2
        genJetEta_2 = events.GenJetEta_2
        genJetMass_2 = events.GenJetMass_2
        LT = events.LT
        HT = events.HT
        metPt = events.MetPt
        WBosonMt = events.WBosonMt
        dPhi = events.DeltaPhi
        nbJets = events.nDeepJetMediumBTag
        zero_b = nbJets == 0
        multi_b = nbJets >= 1
        return locals()
    
    def BaseSelect(self, events, product):
        # set up stuff to fill
        output = self.accumulator.identity()
        selection = processor.PackedSelection()
        size = events.metadata["entrystop"] - events.metadata["entrystart"]
        weights = processor.Weights(size, storeIndividual=self.individal_weights)
        dataset = events.metadata["dataset"]
        output["nEvents"][dataset] = size
        output["nEvents"]["sumAllEvents"] = size
        data = self.config.get_dataset(dataset)
        process = self.config.get_process(dataset)
        sortedJets = ak.mask(events.JetPt, (events.nJet >= 3))
        goodJets = (events.JetPt > 30) & (abs(events.JetEta) < 2.4)
        baselineSelection = ((sortedJets[:, 1] > 80) & (product.LT > 250) & (product.HT > 500) & (ak.num(goodJets) >= 3) & ~(events.IsoTrackVeto))
        # prevent double counting in data
        doubleCounting_XOR = (((events.metadata["PD"] == "isSingleElectron") & events.HLT_EleOr) | ((events.metadata["PD"] == "isSingleMuon") & events.HLT_MuonOr & ~events.HLT_EleOr) | ((events.metadata["PD"] == "isMet") & events.HLT_MetOr & ~events.HLT_MuonOr & ~events.HLT_EleOr))
        self.AddToSelection(selection, "doubleCounting_XOR", doubleCounting_XOR)
        self.AddToSelection((selection), "HLT_Or", events.HLT_MuonOr | events.HLT_MetOr | events.HLT_EleOr)
        self.AddToSelection((selection), "baselineSelection", ak.fill_none(ak.firsts(baselineSelection), False))
        self.AddToSelection((selection), "zero_b", product.zero_b)
        self.AddToSelection((selection), "multi_b", product.multi_b)
        # apply some weights,  MC/data check beforehand
        if not process.is_data:
            weights.add("xSection", process.xSection[13.0].nominal)
        common = ["baselineSelection", "HLT_Or"]  # "HLT_MuonOr", "HLT_MetOr"]
        categories = dict(N0b = ["zero_b"], N1ib = ["multi_b"])
        return locals()
    
    def GetMuonVariables(self, events):
        # leptons variables 
        nMuon = events.nMuon
        leadMuonPt = ak.fill_none(ak.firsts(events.MuonPt[:, 0:1]), 0)
        leadMuonEta = ak.fill_none(ak.firsts(events.MuonEta[:, 0:1]), 0)
        leadMuonPhi = ak.fill_none(ak.firsts(events.MuonPhi[:, 0:1]), 0)
        # MuonMass
        MuonCharge = events.MuonCharge
        MuonPdgId = events.MuonPdgId
        vetoMuon = (events.MuonPt[:, 1:2] > 10) & events.MuonLooseId[:, 1:2]
        genMuonPt = events.GenMuonPt
        genMuonPhi = events.GenMuonPhi
        genMuonEta = events.GenMuonEta
        genMuonMass = events.GenMuonMass
        return locals()

    def GetElectronVariables(self, events):
        # leptons variables 
        nElectron = events.nElectron
        leadElectronPt = ak.fill_none(ak.firsts(events.ElectronPt[:, 0:1]), 0)
        leadElectronEta = ak.fill_none(ak.firsts(events.ElectronEta[:, 0:1]), 0)
        leadElectronPhi = ak.fill_none(ak.firsts(events.ElectronPhi[:, 0:1]), 0)
        # ElectronMass
        electronCharge = events.ElectronCharge
        electronPdgId = events.ElectronPdgId
        vetoElectron = (events.ElectronPt[:, 1:2] > 10) & events.ElectronLooseId[:, 1:2]
        genElectronPt = events.GenElectronPt
        genElectronPhi = events.GenElectronPhi
        genElectronEta = events.GenElectronEta
        genElectronMass = events.GenElectronMass
        return locals()

class ArrayAccumulator(column_accumulator):
    """column_accumulator with delayed concatenate"""

    def __init__(self, value):
        self._empty = value[:0]
        self._value = [value]

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.value)

    def identity(self):
        return self.__class__(self._empty)

    def add(self, other):
        assert self._empty.shape == other._empty.shape
        assert self._empty.dtype == other._empty.dtype
        self._value.extend(v for v in other._value if len(v))

    @property
    def value(self):
        if len(self._value) > 1:
            self._value = [np.concatenate(self._value)]
        return self._value[0]

    def __len__(self):
        return sum(map(len, self._value))


class Histogramer(BaseProcessor, BaseSelection):

    def variables(self):
        return self.config.variables

    def __init__(self, task):
        super().__init__(task)

        self._accumulator["histograms"] = dict_accumulator(
            {var.name: hist.Hist("Counts", self.datasetAxis, self.categoryAxis, 
                                 hist.Bin(var.name, var.x_title, var.binning[0], var.binning[1], var.binning[2])) for var in self.variables()
            })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        out = self.MuonSelect(events)
        weights = out["weights"]
        for var_name in self.variables().names():
            for cat in out["categories"].keys():
                weight = weights.weight()
                # value = out[var_name]
                # generate blank mask for variable values
                mask = np.ones(len(out[var_name]), dtype=bool)
                # combine cuts together: problem, some have None values
                for cut in out["categories"][cat][:1]:  # FIXME
                    cut_mask = ak.to_numpy(out[cut])
                    if type(cut_mask) is np.ma.core.MaskedArray:
                        cut_mask = cut_mask.mask
                    mask = np.logical_and(mask, cut_mask)  # .mask
                values = {}
                values["dataset"] = out["dataset"]
                values["category"] = cat
                # we just want to hist every entry so flatten works since we don't wont to deal with nested array structures
                values[var_name] = ak.flatten(out[var_name][mask], axis=None)
                values["weight"] = weight[mask]
                output["histograms"][var_name].fill(**values)

        # output["n_events"] = len(MetPt)
        return output

    def postprocess(self, accumulator):
        return accumulator


class ArrayExporter(BaseProcessor, BaseSelection):
    output = "*.npy"
    dtype = None
    sep = "_"

    def __init__(self, task, Lepton):
        self.Lepton = Lepton
        super().__init__(task)
        self._accumulator["arrays"] = dict_accumulator()

    def categories(self, select_output):
        selection = select_output.get("selection")
        categories = select_output.get("categories")
        # from IPython import embed;embed()
        return (
            {cat: selection.all(*cuts) for cat, cuts in categories.items()}
            if selection and categories
            else {"all": slice(None)}
        )

    def select(self, events):  # , unc, shift):
        out = super().BaseSelect(events)
        if self.Lepton == "Muon":
            
            out.update(**super().GetMuonVariables(events))  # , unc, shift)
        if self.Lepton == "Electron":
            out.update(**super().ElectronSelect(events))  # , unc, shift)
        dataset = self.GetDataset(events)
        return out

    def process(self, events):
        select_output = self.select(events)  # , unc="nominal", shift=None)
        categories = self.categories(select_output)
        weights = select_output["weights"]
        output = select_output["output"]
        selectedVariables = self.GetSelectedVariables(select_output)
        output["selectedVariables"] = dict_accumulator(
            {category + "_" + select_output["dataset"]: dict_accumulator(
                {key: ArrayAccumulator(array[cut, ...]) for key, array in selectedVariables.items()})
            for category, cut in categories.items()})
        return output

    def postprocess(self, accumulator):
        return accumulator
