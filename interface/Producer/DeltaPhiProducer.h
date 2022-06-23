#ifndef DELTAPHIPRODUCER_H
#define DELTAPHIPRODUCER_H

#include <Susy1LeptonAnalysis/Susy1LeptonSkimmer/interface/Producer/BaseProducer.h>
#include <Susy1LeptonAnalysis/Susy1LeptonSkimmer/interface/Susy1LeptonProduct.h>

#include <cmath>

#include "PhysicsTools/Heppy/interface/Davismt2.h"
#include <TMath.h>

class DeltaPhiProducer : public BaseProducer {
	private:
		double hadronicMt2Cut, leptonicMt2Cut;
	public:
		std::string Name = "DeltaPhiProducer";
		DeltaPhiProducer(const pt::ptree &configTree, const pt::ptree &scaleFactorTree);
		void Produce(DataReader &dataReader, Susy1LeptonProduct &product);
		void EndJob(TFile &file);
};

#endif

