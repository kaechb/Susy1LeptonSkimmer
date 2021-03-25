# DISCLAIMER: currently in development under branch law_introduction

# SusyLeptonAnalysis

TODO: Structure

Creation of the needed conda environment from .yml file (points to my directory, a lot of packages)
(Currently not neccessary when using setup.sh, but can be customized)
```shell
conda environment: conda env create -f law_env.yml
conda activate law_env
```

Beautifying all .py-files from command line (before commiting!): 
```shell
python beautify.py
```

Setup law indexing, autocompletion and task recognition
```shell
law setup: 
source analysis/setup.sh
```

# Current analysis tasks

Writes the config/datasets.py, taken from previously skimmed files
```shell
law run mj.WriteConfigData --local-scheduler --version dev1
```

Creating fileset (loops over all datasets)
```shell
law run mj.WriteFileset --local-scheduler --version dev1
```

Histogram the files and save as coffea.hists
This includes a Hstogram for every category-variable-dataset combination.
```shell
law run mj.CoffeaProcessor --version dev1 --local-scheduler (--processor Histogramer|ArrayExporter)
```
Debug the selection and processor
```shell
law run mj.CoffeaProcessor --version test1 --local-scheduler --processor Histogramer --debug --workflow local
```


Plotting of hists into a pdf, log scale allows for y-axis change
Groups subprocesses into parents, seperate plots for category-variables
```shell
law run mj.PlotCoffeaHists --version dev1 --local-scheduler (--log-scale) (--unblinded) 
```

Save variables into arrays for later computation 
```shell
law run mj.CoffeaProcessor --version dev1 --local-scheduler --processor ArrayExporter
```
