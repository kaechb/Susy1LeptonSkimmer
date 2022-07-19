#ifndef CUTFLOW_H
#define CUTFLOW_H

#include <vector>
#include <string>
#include <functional>
#include <memory>

#include <TFile.h>
#include <TH1F.h>

#include <Susy1LeptonAnalysis/Susy1LeptonSkimmer/interface/Susy1LeptonProduct.h>

class CutFlow{
	private:
		std::shared_ptr<TH1D> hist;
		std::vector<std::function<bool()>> cuts;
		std::vector<std::string> cutNames;

		std::function<bool()> ConstructCut(int &value, const std::string &op, const int &threshold);

	public:
		CutFlow(){}
		CutFlow(TFile &outputFile, const std::string &channel, const std::string &systematic, const std::string &shift);
		void AddCut(const std::string &part, Susy1LeptonProduct& product, const std::string &op, const int &threshold);

		void AddTrigger(const std::vector<int>& triggerIndex, Susy1LeptonProduct &product){
			cuts.insert(cuts.begin(), [&product, triggerIndex](){for(const int& idx : triggerIndex){if(product.triggerValues[idx]) return true;} return false;});
			cutNames.insert(cutNames.begin(), "Trigger");
		}

		void AddMetFilter(Susy1LeptonProduct &product){
			cuts.insert(cuts.begin(), [&product](){for(const bool& passed : product.metFilterValues){if(!passed) return false;} return true;});
			cutNames.insert(cutNames.begin(), "MET Filter");
		}

		bool Passed();
		void Count(){hist->Fill("No cuts", 1);}; // TODO maybe just use GetEntries()?
		void FillCutflow();
		void WriteOutput(){hist->Write(0, TObject::kOverwrite);};
};

#endif

