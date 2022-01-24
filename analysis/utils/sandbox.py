# coding: utf-8

import law


class CMSSWSandboxTask(law.SandboxTask):
    @property
    def sandbox(self):
        return "bash::setup_cmssw.sh"
