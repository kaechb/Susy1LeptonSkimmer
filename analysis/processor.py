from IPython import embed
from utils.Base import *
import coffea

# from coffea.processor import ProcessorABC
# import law
import numpy as np
import uproot4 as up
import awkward1 as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import hist, processor

# import matplotlib.pyplot as plt

from time import time

# register our candidate behaviors
from coffea.nanoevents.methods import candidate

ak.behavior.update(candidate.behavior)

# for jet_ lep_ variables
new_test_file = "/nfs/dust/cms/user/frengelk/Testing/new_TTJets_HT_1200to2500_1.root"

# unchanged syntax
test_file = "/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"

name = test_file.split("/")[-1].split(".")[0]
fileset = {
    "tt": [test_file],
    # "t_t": [new_test_file]
}

hist_inst = Histogramer()


def array_production(fileset, np_path):

    tic = time()

    export_inst = ArrayExporter()

    out = processor.run_uproot_job(
        fileset,
        treename="nominal",
        processor_instance=export_inst,
        executor=processor.iterative_executor,
        # {"schema": NanoAODSchema},
        # maxchunks=4,
        # metadata_cache = {name: [name],}
    )

    np.save(np_path, out["arrays"]["hl"])
    # print(out)

    print(np.round(time() - tic, 4), "s")

    return out


array_production(fileset, "/nfs/dust/cms/user/frengelk/Testing/array.npy")

a = np.load("/nfs/dust/cms/user/frengelk/Testing/array.npy", allow_pickle=True)
print(a)

embed()
