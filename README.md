# SusyLeptonAnalysis

TODO: Structure

Creation of the needed conda environment from .yml file (points to my directory, a lot of packages)
```shell
conda environment: conda env create -f law_env.yml
conda activate law_env
```

Beautifying from command line (beforecommiting!): 
```shell
python beautify.py
```

Setup law indexing, autocompletion and task recognition
```shell
cd analysis
law setup: source setup.sh
```

# Current analysis tasks

Creating fileset (paths hardcoded for now)
```shell
law run mj.WriteFileset --local-scheduler --version dev1
```

Histogram the files and save coffea hists 
```shell
law run mj.CoffeaProcessor --version dev1 --local-scheduler --processor Histogramer
```

Plotting of hists into a pdf, log scale allows for y-axis change
```shell
law run mj.PlotCoffeaHists --version dev1 --local-scheduler (--log-scale) (--unblinded) 
```

Save variables into arrays for later computation 
```shell
law run mj.CoffeaProcessor --version dev1 --local-scheduler --processor ArrayExporter
```
