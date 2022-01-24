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
import math

from utils.sandbox import CMSSWSandboxTask

# define which implemented loaders we want to use
law.contrib.load(
    "numpy", "tasks", "root", "htcondor", "hdf5", "coffea", "matplotlib"
)  # "wlcg",
"""
Collection of different standard task definition, each extending the dfinitions
-Basetask defines basic functions
-Campaign task adds the config instance needed everywhere, loads it from the respective config directory
-Analysis task defines the namespace TODO: get it from $PATH
-Config, Shift, Dataset task present in law example, not really needed rn
-HTCondorwrokflow for task submission using HTCondorworkflow
-CMSSW to set up a CMSSW environment
"""


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
    # analysis_id = "mj"
    analysis_id = "0b"
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


class DNNTask(ConfigTask):
    """
    define parameters here for all relevant tasks
    """

    channel = luigi.Parameter(default="0b", description="channel to train on")
    epochs = luigi.IntParameter(default=100)
    batch_size = luigi.IntParameter(default=10000)
    learning_rate = luigi.FloatParameter(default=0.01)
    debug = luigi.BoolParameter(default=False)
    n_layers = luigi.IntParameter(default=3)
    n_nodes = luigi.IntParameter(default=256)
    dropout = luigi.FloatParameter(default=0.2)

    def __init__(self, *args, **kwargs):
        super(DNNTask, self).__init__(*args, **kwargs)


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


class DatasetTask(ConfigTask):  # ShiftTask

    dataset = luigi.Parameter(
        default="TTJets_sl_fromt", description="the dataset name, default: "
    )

    # @classmethod
    # def modify_param_values(cls, params):
    # if params["shift"] == "nominal":
    # return params

    # # shift known to config?
    # config_inst = od.Config.get_instance(cls.config)
    # # if params["shift"] not in config_inst.shifts:
    # # raise Exception(
    # # "shift {} unknown to config {}".format(params["shift"], config_inst)
    # # )

    # # # check if the shift is known to the task or dataset
    # # dataset_inst = od.Dataset.get_instance(params["dataset"])
    # # if params["shift"] in cls.shifts or params["shift"] in dataset_inst.info:
    # # params["effective_shift"] = params["shift"]

    # return params

    def __init__(self, *args, **kwargs):
        super(DatasetTask, self).__init__(*args, **kwargs)

        # store the dataset instance and the dataset info instance that
        # corresponds to the shift
        self.dataset_inst = self.config_inst.get_dataset(self.dataset)
        # self.dataset_info_inst = self.dataset_inst.get_info(
        # self.shift_inst.name
        # if self.shift_inst.name in self.dataset_inst.info
        # else "nominal"
        # )

        # also, when there is only one linked process in the current dataset,
        # store it
        if len(self.dataset_inst.processes) == 1:
            self.process_inst = list(self.dataset_inst.processes.values())[0]
        else:
            self.process_inst = None

    # @property
    # def store_parts(self):
    # parts = super(DatasetTask, self).store_parts
    # # insert the dataset name right before the shift
    #
    # parts = parts[:-1] + (self.dataset, parts[-1])
    # return parts


# Syntax setting of condor output
# class HTCondorJobManagerRWTH(law.htcondor.HTCondorJobManager):
## @classmethod
# def map_status(cls, status_flag):
# # map status hold ("5", "H") and suspended ("7") to
# # cls.PENDING to avoid resubmission
# if status_flag in ("0", "1", "5", "7", "U", "I", "H"):
# return cls.PENDING
# elif status_flag in ("2", "R"):
# return cls.RUNNING
# elif status_flag in ("4", "C"):
# return cls.FINISHED
# elif status_flag in ("6", "E"):
# return cls.FAILED
# else:
# return cls.FAILED


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    debug = luigi.BoolParameter()
    """
    Batch systems are typically very heterogeneous by design, and so is HTCondor. Law does not aim
    to "magically" adapt to all possible HTCondor setups which would certainly end in a mess.
    Therefore we have to configure the base HTCondor workflow in law.contrib.htcondor to work with
    the CERN HTCondor environment. In most cases, like in this example, only a minimal amount of
    configuration is required.
    """

    def create_branch_map(self):
        # trivial branch map: one branch per file
        # from IPython import embed;embed()
        n = self.config_inst.datasets.len()
        # dividde workload
        n = n * 1
        if self.debug:
            n = 1
        return list(range(n))

        # return {i: i for i in range(n)}

    def htcondor_post_submit_delay(self):
        return self.poll_interval * 60

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path())

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return law.util.rel_path("$ANALYSIS_BASE", "htcondor_bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        # render_variables are rendered into all files sent with a job
        config.render_variables["analysis_base"] = os.getenv("ANALYSIS_BASE")
        # copy the entire environment
        config.custom_content.append(("getenv", "True"))
        # config.custom_content.append(("SCHEDD_NAME", "bird-htc-sched14.desy.de"))
        config.custom_content.append(("universe", "vanilla"))
        # require more RAM on CPU
        # config.custom_content.append(("request_cpus", "1"))
        config.custom_content.append(("request_memory", "15000"))
        config.custom_content.append(("+RequestRuntime = 86400"))
        # config.custom_content.append(("Request_GPUs", "0"))
        # config.custom_content.append(("Request_GpuMemory", "0"))

        # condor logs
        # if self.htcondor_logs:
        config.stdout = "out.txt"
        config.stderr = "err.txt"
        config.log = "log.txt"
        # from IPython import embed;embed()
        return config

    def htcondor_use_local_scheduler(self):
        return True

    # def htcondor_create_job_manager(self, **kwargs):
    # kwargs = law.util.merge_dicts(
    # self.htcondor_job_manager_defaults, kwargs)
    # return HTCondorJobManagerRWTH(**kwargs)


class InstallCMSSWCode(CMSSWSandboxTask, law.tasks.RunOnceTask, DatasetTask):

    clean = luigi.BoolParameter(
        default=False, description="run 'scram b clean' before installing"
    )
    cores = luigi.IntParameter(
        default=1, description="the number of cores for compilation"
    )

    # task_namespace = "{}".format(os.environ["DHA_ANALYSIS_ID"])

    # version = None

    @law.decorator.notify
    def run(self):
        import os
        import shutil

        # copy the current cmssw code to the CMSSW_BASE directory
        CMSSW_BASE = "/nfs/dust/cms/user/frengelk/Code/cmssw/CMSSW_10_2_13"
        """
        src = os.path.join(os.environ["DHA_BASE"], "cmssw", subsystem)
        dst = os.path.join(os.environ["CMSSW_BASE"], "src", subsystem)
        if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            self.publish_message("created package {}".format(dst))
        """

        # build the command
        cmd = "scram b -j {}".format(self.cores)
        if self.clean:
            cmd = "scram b clean; {}".format(cmd)

        # run the command
        code = law.util.interruptable_popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            # cwd=os.path.join(os.environ["CMSSW_BASE"], "src"),
            cwd=os.path.join(CMSSW_BASE),
        )[0]
        if code != 0:
            raise Exception("scram build failed")

        # mark as complete
        self.mark_complete()
