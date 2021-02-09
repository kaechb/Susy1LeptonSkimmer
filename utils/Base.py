import coffea
#from coffea.processor import ProcessorABC
#import law
import numpy
import uproot4 as up
import awkward1 as ak
from coffea.nanoevents import NanoEventsFactory,  NanoAODSchema
from coffea import hist, processor
from coffea.hist.hist_tools import DenseAxis, Hist

from coffea.processor.accumulator import (
    dict_accumulator,
    defaultdict_accumulator,
    column_accumulator,
    set_accumulator,
)

from . import config as cfg

# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)


class BaseProcessor(processor.ProcessorABC):
    #individal_weights = False
    #jes_shifts = False
    #dataset_shifts = False

    def __init__(self):
        #self.publish_message = task.publish_message if task.debug else None
        #self.config = task.config_inst
        #self.corrections = task.load_corrections()

        self.dataset_axis = hist.Cat("dataset", "Primary dataset")
        self.dataset_shift_axis = hist.Cat("dataset_shift", "Dataset shift")
        self.category_axis = hist.Cat("category", "Category selection")
        self.syst_axis = hist.Cat("systematic", "Shift of systematic uncertainty")
        self._accumulator = dict_accumulator(
            n_events=defaultdict_accumulator(float),
            sum_gen_weights=defaultdict_accumulator(float),
            object_cutflow=defaultdict_accumulator(int),
            cutflow=hist.Hist(
                "Counts",
                self.dataset_axis,
                self.category_axis,
                hist.Bin("cutflow", "Cut index", 10, 0, 10),
            ),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def get_dataset(self, events):
        return self.config.get_dataset(events.metadata["dataset"])

    def get_dataset_shift(self, events):
        return events.metadata["dataset"][1]

    def get_lfn(self, events):
        ds = self.get_dataset(events)
        fn = events.metadata["filename"].rsplit("/", 1)[-1]

        for lfn in ds.info[self.get_dataset_shift(events)].aux["lfns"]:
            if lfn.endswith(fn):
                return lfn
        else:
            raise RuntimeError("could not find original LFN for: %s" % events.metadata["filename"])

    def get_pu_key(self, events):
        ds = self.get_dataset(events)
        if ds.is_data:
            return "data"
        else:
            lfn = self.get_lfn(events)
            for name, hint in ds.campaign.aux.get("pileup_lfn_scenario_hint", {}).items():
                if hint in lfn:
                    return name
            else:
                return "MC"

class Histogramer(BaseProcessor):

    def __init__(self):
        super().__init__()

        self._accumulator["histograms"] = dict_accumulator(
            {
                var[0]: hist.Hist(
                    "Counts",
                    self.dataset_axis,
                    #self.category_axis,
                    #self.syst_axis,
                    hist.Bin(
                        var[0],
                        var[1],
                        var[2][0],
                        var[2][1],
                        var[2][2],
                    ),
                )
                for var in cfg.variables().values()
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        #set up stuff to fill
        output = self.accumulator.identity()
        selection = processor.PackedSelection()

        #from IPython import embed;embed()

        #branches = file.get("nominal")
        dataset= events.dataset

        # event variables
        METPt = events.METPt
        W_mt = events.WBosonMt

        #leptons variables
        n_leptons=events.nLepton
        lep_pt = events.LeptonPt[:,0]
        tight_lep=events.LeptonTightId[:,0]

        leptons = ak.zip({
             "pt": events.LeptonPt,
             "eta": events.LeptonEta,
             "phi": events.LeptonPhi,
             "mass": events.LeptonMass,
             "charge": events.LeptonCharge,
         }, with_name="PtEtaPhiMCandidate")

        # lep selection
        lep_selection =(
            (n_leptons == 1)
            &(tight_lep)
            &(lep_pt > 10)
            )

        selection.add("lep_selection", ak.to_numpy(lep_selection))

        #jet variables
        n_jets =events.nJet
        n_btags = events.nMediumDFBTagJet
        jet_mass_1 = events.JetMass[:,0]

        Dphi= events.DeltaPhi
        LT = events.LT
        HT=events.HT
        sorted_jets = ak.sort(events.JetPt, ascending=False)

        baseline_selection= (
            (lep_pt > 25)
            #&veto lepton > 10
            #&No isolated track with p T ≥ 10 GeV and M T2 < 60 GeV (80 GeV) for hadronic (leptonic)
            &(sorted_jets[:,0] > 80)
            &(LT>250)
            &(HT>500)
            &(n_jets>=3)
            )

        zero_b = (n_btags==0)
        multi_b = (n_btags>=1)
        selection.add("baseline", ak.to_numpy(baseline_selection))


        """
        output["met"].fill(
            dataset=dataset,
            met=met[lep_selection],
        )
	    output["lep_pt"].fill(
            dataset=dataset,
            lep_pt=lep_pt[lep_selection],
        )

	    output["WBosonMt"].fill(
            dataset=dataset,
            WBosonMt=W_mt[lep_selection],
        )
	    output["jet_mass_1"].fill(
            dataset=dataset,
            jet_mass_1=jet_mass_1[lep_selection],
        )
        """


        #from IPython import embed;embed()

        for key in cfg.variables().keys():
            values={}
            values["dataset"]=dataset
            values[key]=eval(key)
            #weight = weights.weight()[cut]
            #values["weight"] = weight
            output["histograms"][key].fill(**values)

        #output["n_events"] = len(METPt)

	    # test
        return output

    def postprocess(self, accumulator):
        return accumulator
