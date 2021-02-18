import os
import law
import uproot as up
from luigi import Parameter, BoolParameter
from tasks.basetasks import *
import json


class DownloadFiles(ConfigTask):


class DownloadFilesWrapper(CampaignTask, law.WrapperTask):


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
        test_file = "/nfs/dust/cms/user/frengelk/Testing/TTJets_HT_1200to2500_1.root"

        fileset = {
            "tt": [test_file],
        }

        with open(self.output(), w) as file:
            json.dump(fileset, file)
