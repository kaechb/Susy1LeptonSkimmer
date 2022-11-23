#include <Susy1LeptonAnalysis/Susy1LeptonSkimmer/interface/Producer/FastSimProducer.h>

FastSimProducer::FastSimProducer(const pt::ptree &configTree, const pt::ptree &scaleFactorTree, std::string eraSelector, TFile &outputFile) {
	Name = "FastSimProducer";
	std::string cmsswBase = std::getenv("CMSSW_BASE");

	// Open root files stored in root files
	electronSfFile = (TFile*) TFile::Open((cmsswBase + "/src/Susy1LeptonAnalysis/Susy1LeptonSkimmer/data/fastsim/" + eraSelector + "/" + scaleFactorTree.get<std::string>("Electron." + eraSelector + ".FastSim.Path")).c_str());
	muonSfFile     = (TFile*) TFile::Open((cmsswBase + "/src/Susy1LeptonAnalysis/Susy1LeptonSkimmer/data/fastsim/" + eraSelector + "/" + scaleFactorTree.get<std::string>("Muon." + eraSelector + ".ScaleFactor.FastSim.Path")).c_str());

	// electron keys
	// https://twiki.cern.ch/twiki/bin/view/CMS/SUSLeptonSF#Electrons_FullSim_FastSim
	electronVetoSf     = std::make_shared<TH2F>(*(TH2F*)electronSfFile->Get("CutBasedVetoNoIso94XV2_sf"));
	electronTightSf    = std::make_shared<TH2F>(*(TH2F*)electronSfFile->Get("CutBasedTightNoIso94XV2_sf"));
	electronVetoMvaSf = std::make_shared<TH2F>(*(TH2F*)electronSfFile->Get("MVAVLooseTightIP2DMini4_sf")); // not sure if this is correct
	electronTightMvaSf = std::make_shared<TH2F>(*(TH2F*)electronSfFile->Get("ConvIHit0_sf"));

	//Muon keys
	muonLooseSf      = std::make_shared<TH2F>(*(TH2F*)muonSfFile->Get("miniIso04_LooseId_sf"));
	muonMediumSf      = std::make_shared<TH2F>(*(TH2F*)muonSfFile->Get("miniIso02_MediumId_sf"));
}

void FastSimProducer::Produce(DataReader &dataReader, Susy1LeptonProduct &product) {
	// Lepton Scale Factors between FullSim and FastSim
	for (int iMuon = 0; iMuon < product.nMuon; iMuon++) {
		const float &muonPt  = product.muonPt[iMuon];
		const float &muonEta = std::abs(product.muonEta[iMuon]);
		product.muonLooseFastSf[iMuon]  = Utility::Get2DWeight(muonEta, muonPt, static_cast<TH2F*>(muonLooseSf.get())); // function needs raw pointer, shared_ptr()->get() returns it
		product.muonMediumFastSf[iMuon] = Utility::Get2DWeight(muonEta, muonPt, static_cast<TH2F*>(muonMediumSf.get()));

		const float &looseFastSfErr  = Utility::Get2DWeightErr(muonEta, muonPt, static_cast<TH2F*>(muonLooseSf.get()));
		const float &mediumFastSfErr = Utility::Get2DWeightErr(muonEta, muonPt, static_cast<TH2F*>(muonMediumSf.get()));

		product.muonLooseFastSfUp[iMuon]  = product.muonLooseFastSf[iMuon]  + looseFastSfErr;
		product.muonMediumFastSfUp[iMuon] = product.muonMediumFastSf[iMuon] + mediumFastSfErr;

		product.muonLooseFastSfDown[iMuon]  = product.muonLooseFastSf[iMuon]  - looseFastSfErr;
		product.muonMediumFastSfDown[iMuon] = product.muonMediumFastSf[iMuon] - mediumFastSfErr;
	}

	for (int iElectron = 0; iElectron < product.nElectron; iElectron++) {
		const float &electronPt  = product.electronPt[iElectron];
		const float &electronEta = std::abs(product.electronEta[iElectron]);
		product.electronVetoFastSf[iElectron]     = Utility::Get2DWeight(electronEta, electronPt, static_cast<TH2F*>(electronVetoSf.get()));
		product.electronTightFastSf[iElectron]    = Utility::Get2DWeight(electronEta, electronPt, static_cast<TH2F*>(electronTightSf.get()));
		product.electronVetoMvaFastSf[iElectron]  = Utility::Get2DWeight(electronEta, electronPt, static_cast<TH2F*>(electronVetoMvaSf.get()));
		product.electronTightMvaFastSf[iElectron] = Utility::Get2DWeight(electronEta, electronPt, static_cast<TH2F*>(electronTightMvaSf.get()));

		const float &vetoFastSfErr     = Utility::Get2DWeightErr(electronEta, electronPt, static_cast<TH2F*>(electronVetoSf.get()));
		const float &tightFastSfErr    = Utility::Get2DWeightErr(electronEta, electronPt, static_cast<TH2F*>(electronTightSf.get()));
		const float &vetoMvaFastSfErr  = Utility::Get2DWeightErr(electronEta, electronPt, static_cast<TH2F*>(electronVetoMvaSf.get()));
		const float &tightMvaFastSfErr = Utility::Get2DWeightErr(electronEta, electronPt, static_cast<TH2F*>(electronTightMvaSf.get()));

		product.electronVetoFastSfUp[iElectron]    = product.electronVetoFastSf[iElectron]     + vetoFastSfErr;
		product.electronTightFastSfUp[iElectron]   = product.electronTightFastSf[iElectron]    + tightFastSfErr;
		product.electronVetoMvaFastSfUp[iElectron] = product.electronVetoMvaFastSf[iElectron]  + vetoMvaFastSfErr;
		product.electronTightMvaFastSfUp[iElectron]= product.electronTightMvaFastSf[iElectron] + tightMvaFastSfErr;

		product.electronVetoFastSfDown[iElectron]    = product.electronVetoFastSf[iElectron]     - vetoFastSfErr;
		product.electronTightFastSfDown[iElectron]   = product.electronTightFastSf[iElectron]    - tightFastSfErr;
		product.electronVetoMvaFastSfDown[iElectron] = product.electronVetoMvaFastSf[iElectron]  - vetoMvaFastSfErr;
		product.electronTightMvaFastSfDown[iElectron]= product.electronTightMvaFastSf[iElectron] - tightMvaFastSfErr;
	}

	//TODO
	// https://twiki.cern.ch/twiki/bin/view/CMS/SUSRecommendations18#Cleaning_up_of_fastsim_jets_from
}

void FastSimProducer::EndJob(TFile &file) {
	electronSfFile->Close();
	muonSfFile->Close();
}