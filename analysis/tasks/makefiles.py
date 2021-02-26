import os
import law
import uproot as up
from luigi import Parameter, BoolParameter
from tasks.basetasks import *
import json

"""
class DownloadFiles(ConfigTask):
class DownloadFilesWrapper(CampaignTask, law.WrapperTask):
"""


class WriteFileset(AnalysisTask):

    # def requires(self):
    # te

    def output(self):
        return self.local_target("fileset.json")

    def run(self):
        # make the output directory
        out = self.output().parent
        out.touch()

        # unchanged syntax
        # test_file = "/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"

        fileset = {}

        for dat in self.config_inst.datasets:
            fileset.update(
                {
                    dat.name: dat.keys,
                }
            )

        # from IPython import embed;embed()

        with open(self.output().path, "w") as file:
            json.dump(fileset, file)
