import coffea
#from coffea.processor import ProcessorABC
import law
import numpy
import uproot4 as up
import awkward1 as ak
from coffea.nanoevents import NanoEventsFactory,  NanoAODSchema
from coffea import hist, processor
import matplotlib.pyplot as plt

# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

print("hello world")

# for jet_ lep_ variables
new_test_file = "~/nfs/dust/cms/user/frengelk/Testing/new_TTJets_HT_1200to2500/new_TTJets_HT_1200to2500_1.root"

# unchanged syntax
test_file = "/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"

name = test_file.split('/')[-1].split('.')[0]
fileset = {"tt" : [test_file],
          # "t_t": [new_test_file]
           }

file = up.open(test_file)

#class BaseProcessor(ProcessorABC):
#    output = "data.coffea"


class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "sumw": processor.defaultdict_accumulator(float),
            "met": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("met", "MET$_{pt}$ [GeV]", 150, 0, 750),
            ),
            "lep_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("lep_pt", "Lepton$_{pt}$ [GeV]", 100, 0, 500),
            ),
            "WBosonMt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("WBosonMt", "W$_{mt}$ [GeV]", 200, 0,1000),
            ),
            "jet_mass_1": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("jet_mass_1", "Jet$_{mass}^{1}$ [GeV]", 100, 0, 500),
            ),

        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        #set up stuff to fill
        output = self.accumulator.identity()
        selection = processor.PackedSelection()


        from IPython import embed;embed()

        #branches = file.get("nominal")
        dataset= events.dataset

	# b.arrays()[b"out"][0](namedecode='utf-8')

        #events = NanoEventsFactory.from_root(
        #    file,
        #    treepath="nominal",
        #    #entry_stop=10000,
        #    metadata={"dataset": dataset},
        #    schemaclass=BaseSchema,
        #).events()

        # event variables
        met = events.METPt
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

        #cut = (ak.num(muons) == 2) & (ak.sum(muons.charge) == 0)
        # add first and second muon in every event together
        #dimuon = muons[cut][:, 0] + muons[cut][:, 1]

        #from IPython import embed;embed()

        output["sumw"][dataset] += len(met)
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
        # test
        return output

    def postprocess(self, accumulator):
        return accumulator

# simple call
#proc = MyProcessor()
#out = proc.process(file)

out = processor.run_uproot_job(
    fileset,
    treename="nominal",
    processor_instance=MyProcessor(),
    executor=processor.iterative_executor,
    #{"schema": NanoAODSchema},
    #maxchunks=4,
    #metadata_cache = {name: [name],}
)


print("lets plot")

from IPython import embed; embed()

for key in out.keys():
    if key=="sumw":
       continue

    fig = plt.figure()
    coffea.hist.plot1d(out[key])
    fig.suptitle('{0}, after cuts'.format(key))
    fig.savefig('/nfs/dust/cms/user/frengelk/figures/{0}_after_cut.png'.format(key))
    fig.clear()




#fig = plt.figure()
#coffea.hist.plot1d(out["lep_pt"])
#fig.suptitle('lep_pt_after_cut')
#fig.savefig('lep_pt_after_cut.png')
#fig.clear()


#from IPython import embed; embed()
