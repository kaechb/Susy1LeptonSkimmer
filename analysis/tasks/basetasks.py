# coding: utf-8
"""
Base task classes that are used across the analysis.
"""

__all__ = ["AnalysisTask", "ConfigTask", "ShiftTask", "DatasetTask"]

import os

import luigi
import law
import order as od
import importlib

law.contrib.load(
    "numpy", "tasks", "root", "slack", "telegram", "wlcg", "htcondor", "hdf5", "coffea"
)


class BaseTask(law.Task):
    version = luigi.Parameter(default="dev1", description="version of current workflow")

    # notify = law.telegram.NotifyTelegramParameter()

    # exclude_params_req = {"notify"}
    # exclude_params_branch = {"notify"}
    # exclude_params_workflow = {"notify"}
    # output_collection_cls = law.SiblingFileCollection
    # workflow_run_decorators = [law.decorator.notify]
    # message_cache_size = 20

    def local_path(self, *path):
        parts = [str(p) for p in self.store_parts() + path]
        return os.path.join("/nfs/dust/cms/group/susy-desy/Susy1Lepton", *parts)
        # return os.path.join(os.environ["DHA_STORE"], *parts)

    # def wlcg_path(self, *path):
    # parts = [str(p) for p in self.store_parts() + path]
    # return os.path.join(*parts)

    def local_target(self, *args):
        cls = law.LocalFileTarget if args else law.LocalDirectoryTarget
        return cls(self.local_path(*args))

    def local_directory_target(self, *args):
        return law.LocalDirectoryTarget(self.local_path(*args))

    # def wlcg_target(self, *args, **kwargs):
    # cls = law.wlcg.WLCGFileTarget if args else law.wlcg.WLCGDirectoryTarget
    # return cls(self.wlcg_path(*args), **kwargs)


class CampaignTask(BaseTask):
    year = luigi.Parameter(description="Year", default="2016")
    config = luigi.Parameter(default="SUSY_1lep_ML", description="current analysis")
    analysis_choice = "common"

    def __init__(self, *args, **kwargs):
        super(CampaignTask, self).__init__(*args, **kwargs)
        self.campaign_name = "Run2_pp_13TeV_{}".format(self.year)
        self.campaign_inst = importlib.import_module(
            "config.{}".format(self.campaign_name)
        ).campaign

    def store_parts(self):
        parts = (self.analysis_choice, self.campaign_name, self.__class__.__name__)
        if self.version is not None:
            parts += (self.version,)
        return parts


class AnalysisTask(CampaignTask):

    analysis_id = "mj"
    # luigi.Parameter(
    # default="mj",
    # description="type of analysis, start with mj",
    # ) # os.environ["DHA_ANALYSIS_ID"]

    task_namespace = "{}".format(analysis_id)

    analysis_choice = analysis_id

    def __init__(self, *args, **kwargs):
        super(AnalysisTask, self).__init__(*args, **kwargs)
        self.analysis_inst = self._import("config.analysis").analysis
        self.config_inst = self.analysis_inst.get_config(self.campaign_name)

    def _import(self, *parts):
        return importlib.import_module(".".join((self.analysis_choice,) + parts))

    @classmethod
    def modify_param_values(cls, params):
        return params


class ConfigTask(AnalysisTask):
    def __init__(self, *args, **kwargs):
        super(ConfigTask, self).__init__(*args, **kwargs)
        self.config_inst = self.analysis_inst.get_config(self.campaign_name)

    def store_parts(self):
        parts = (self.analysis_choice, self.campaign_name, self.__class__.__name__)
        if self.version is not None:
            parts += (self.version,)
        return parts


class ShiftTask(ConfigTask):

    shift = luigi.Parameter(
        default="nominal",
        significant=False,
        description="systematic shift to " "apply, default: nominal",
    )
    effective_shift = luigi.Parameter(default="nominal")

    shifts = set()

    exclude_params_index = {"effective_shift"}
    exclude_params_req = {"effective_shift"}
    exclude_params_sandbox = {"effective_shift"}

    @classmethod
    def modify_param_values(cls, params):
        if params["shift"] == "nominal":
            return params

        # shift known to config?
        config_inst = od.Config(cls.config)
        if params["shift"] not in config_inst.shifts:
            raise Exception(
                "shift {} unknown to config {}".format(params["shift"], config_inst)
            )

        # check if the shift is known to the task
        if params["shift"] in cls.shifts:
            params["effective_shift"] = params["shift"]

        return params

    def __init__(self, *args, **kwargs):
        super(ShiftTask, self).__init__(*args, **kwargs)

        # store the shift instance
        self.shift_inst = self.config_inst.get_shift(self.effective_shift)

    @property
    def store_parts(self):
        return super(ShiftTask, self).store_parts + (self.effective_shift,)


class DatasetTask(ShiftTask):

    dataset = luigi.Parameter(
        default="singleTop", description="the dataset name, default: " "singleTop"
    )

    @classmethod
    def modify_param_values(cls, params):
        if params["shift"] == "nominal":
            return params

        # shift known to config?
        config_inst = od.Config.get_instance(cls.config)
        if params["shift"] not in config_inst.shifts:
            raise Exception(
                "shift {} unknown to config {}".format(params["shift"], config_inst)
            )

        # check if the shift is known to the task or dataset
        dataset_inst = od.Dataset.get_instance(params["dataset"])
        if params["shift"] in cls.shifts or params["shift"] in dataset_inst.info:
            params["effective_shift"] = params["shift"]

        return params

    def __init__(self, *args, **kwargs):
        super(DatasetTask, self).__init__(*args, **kwargs)

        # store the dataset instance and the dataset info instance that
        # corresponds to the shift
        self.dataset_inst = self.config_inst.get_dataset(self.dataset)
        self.dataset_info_inst = self.dataset_inst.get_info(
            self.shift_inst.name
            if self.shift_inst.name in self.dataset_inst.info
            else "nominal"
        )

        # also, when there is only one linked process in the current dataset,
        # store it
        if len(self.dataset_inst.processes) == 1:
            self.process_inst = list(self.dataset_inst.processes.values())[0]
        else:
            self.process_inst = None

    @property
    def store_parts(self):
        parts = super(DatasetTask, self).store_parts
        # insert the dataset name right before the shift
        parts = parts[:-1] + (self.dataset, parts[-1])
        return parts

    def create_branch_map(self):
        # trivial branch map: one branch per file
        return {i: i for i in range(self.dataset_info_inst.n_files)}
