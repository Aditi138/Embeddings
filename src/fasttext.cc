/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"

#include <fenv.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

namespace fasttext {

void FastText::getVector(Vector& vec, const std::string& word) {
  const std::vector<int32_t>& ngrams = dict_->getNgrams(word);
  vec.zero();
  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
	vec.addRow(*input_, *it);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}
/*
void FastText::getSubVector(Vector& vec, Vector& lemmavec, Vector& morphvec,Vector& ipavec,std::map<std::string,int>& mapOfSubWords, const int32_t& word,std::ofstream& ofs, std::ofstream& ofl, std::ofstream& ofm,std::ofstream& ofi){
   const std::vector<subentry>& subngrams = dict_->getSubNgrams(word);
   vec.zero();
   lemmavec.zero();
   morphvec.zero();
   ipavec.zero();
   for(int32_t it = 0;it !=subngrams.size(); it++){
	if(mapOfSubWords.find(subngrams[it].subword) == mapOfSubWords.end()){
 		//std::cout << subngrams[it].subword << std::endl;
		vec.zero();
		mapOfSubWords.insert(std::make_pair(subngrams[it].subword, 1));
  		vec.addRow(*input_,subngrams[it].id);
    		ofs << subngrams[it].subword << " " << vec << std::endl;
 	}
    }
   const std::vector<subentry>& subipangrams = dict_->getSubIpaNgrams(word);
   for(int32_t it = 0;it !=subipangrams.size(); it++){
	if(mapOfSubWords.find(subipangrams[it].subword) == mapOfSubWords.end()){
 		//std::cout << subngrams[it].subword << std::endl;
		ipavec.zero();
		mapOfSubWords.insert(std::make_pair(subipangrams[it].subword, 1));
  		ipavec.addRow(*input_,subipangrams[it].id);
    		ofi << subipangrams[it].subword << " " << ipavec << std::endl;
 	}
    }
    const subentry subl = dict_->getLemma(word);
    const std::vector<subentry> subm = dict_->getMorph(word);
    if(mapOfSubWords.find(subl.subword) == mapOfSubWords.end()){
	lemmavec.zero();
	mapOfSubWords.insert(std::make_pair(subl.subword, 1));
  	lemmavec.addRow(*input_,subl.id);
    	ofl << subl.subword << " " << lemmavec << std::endl;
 	}
    else{
       

}
    for(int32_t m=0;m<subm.size();m++){
    subentry e= subm[m];
    if(mapOfSubWords.find(e.subword) == mapOfSubWords.end()){
	morphvec.zero();
	mapOfSubWords.insert(std::make_pair(e.subword, 1));
  	morphvec.addRow(*input_,e.id);
    	ofm << e.subword << " " << morphvec << std::endl;
 	}
      }

}*/

void FastText::getSubVector(Vector& vec, Vector& lemmavec, Vector& morphvec,Vector& ipavec,std::map<int32_t,std::string>& mapOfSubWords, const int32_t& word,std::ofstream& ofs, std::ofstream& ofl, std::ofstream& ofm,std::ofstream& ofi){
   const std::vector<subentry>& subngrams = dict_->getSubNgrams(word);
   vec.zero();
   lemmavec.zero();
   morphvec.zero();
   ipavec.zero();
   for(int32_t it = 0;it !=subngrams.size(); it++){
	int32_t h = dict_->getHash(subngrams[it].subword);
	if(mapOfSubWords.find(h) == mapOfSubWords.end()){  //First use of that hash
 		//std::cout << subngrams[it].subword << std::endl;
		vec.zero();
		mapOfSubWords.insert(std::make_pair(h,subngrams[it].subword));
  		vec.addRow(*input_, h);
    		ofs << subngrams[it].subword << " " << vec << std::endl;
 	}
/*else{
   		std::cout << "Hash already used ngram" << std::endl;
	}*/
 }
   const std::vector<subentry>& subipangrams = dict_->getSubIpaNgrams(word);
   for(int32_t it = 0;it !=subipangrams.size(); it++){
	int32_t h = dict_->getHash(subipangrams[it].subword);
	if(mapOfSubWords.find(h) == mapOfSubWords.end()){
 		//std::cout << subngrams[it].subword << std::endl;
		ipavec.zero();
		mapOfSubWords.insert(std::make_pair(h,subipangrams[it].subword));
  		ipavec.addRow(*input_,h);
    		ofi << subipangrams[it].subword << " " << ipavec << std::endl;
 	}
	/*else{
		 std::cout << "Hash already used by ipangram  " << std::endl;
	}*/
}
    const subentry subl = dict_->getLemma(word);
    const std::vector<subentry> subm = dict_->getMorph(word); 
    int32_t h = dict_->getHash(subl.subword);
   if(subl.subword != "") {
   	if(mapOfSubWords.find(h) == mapOfSubWords.end()){
		lemmavec.zero();
		mapOfSubWords.insert(std::make_pair(h,subl.subword));
  		lemmavec.addRow(*input_,h);
    		ofl << subl.subword << " " << lemmavec << std::endl;
		}
    	/*else{
		std::cout << "Hash already used by lemma: " << std::endl;
		}*/
	}
    for(int32_t m=0;m<subm.size();m++){
    subentry e= subm[m];
  //  std::cout << e.subword << std::endl;
    h  = dict_->getHash(e.subword);
    if(mapOfSubWords.find(h) == mapOfSubWords.end()){
	morphvec.zero();
	mapOfSubWords.insert(std::make_pair(h,e.subword));
  	morphvec.addRow(*input_,h);
    	ofm << e.subword << " " << morphvec << std::endl;
 	}
    /*else {
	 std::cout << "Hash already used by morph: " << std::endl; 
	}*/
      }

}

void FastText::saveVectors() {
  std::cout << "Saving Vectors" << std::endl;
  std::ofstream ofs(args_->output + ".vec");
  std::ofstream ofsub(args_->suboutput + ".vec");
  std::ofstream oflemma(args_->lemmaoutput + ".vec");
  std::ofstream ofmorph(args_->morphoutput + ".vec");
  std::ofstream ofi(args_->ipaoutput + ".vec");
  if (!ofs.is_open()) {
    std::cout << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!ofsub.is_open()) {
    std::cout << "Error opening file for saving subword vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  Vector subvec(args_->dim);
  Vector sublemma(args_->dim);
  Vector submorph(args_->dim);
  Vector subipa(args_->dim);
  std::map<int32_t,std::string> mapOfSubWords;
 
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
   //std::cout << i << std::endl; 
   getVector(vec, word);
   getSubVector(subvec,sublemma,submorph,subipa, mapOfSubWords,i,ofsub,oflemma,ofmorph,ofi);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveModel() {
  std::ofstream ofs(args_->output + ".bin", std::ofstream::binary);
  if (!ofs.is_open()) {
    std::cerr << "Model file cannot be opened for saving!" << std::endl;
    exit(EXIT_FAILURE);
  }
  args_->save(ofs);
  dict_->save(ofs);
  input_->save(ofs);
  output_->save(ofs);
  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  loadModel(ifs);
  ifs.close();
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  args_->load(in);
  dict_->load(in);
  input_->load(in);
  output_->load(in);
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::printInfo(real progress, real loss) {
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst = real(tokenCount) / t;
  real lr = args_->lr * (1.0 - progress);
  int eta = int(t / progress * (1 - progress) / args_->thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cout << std::fixed;
  std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cout << "  lr: " << std::setprecision(6) << lr;
  std::cout << "  loss: " << std::setprecision(6) << loss;
  std::cout << "  eta: " << etah << "h" << etam << "m ";
  std::cout << std::flush;
}

void FastText::supervised(Model& model, real lr,
                          const std::vector<int32_t>& line,
                          const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

void FastText::cbow(Model& model, real lr,
                    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::test(std::istream& in, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF) {
    dict_->getLine(in, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << ": " << precision / nlabels << std::endl;
  std::cout << "Number of examples: " << nexamples << std::endl;
}

void FastText::predict(std::istream& in, int32_t k,
                       std::vector<std::pair<real,std::string>>& predictions) const {
  std::vector<int32_t> words, labels;
  dict_->getLine(in, words, labels, model_->rng);
  dict_->addNgrams(words, args_->wordNgrams);
  if (words.empty()) return;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());
  std::vector<std::pair<real,int32_t>> modelPredictions;
  model_->predict(words, k, modelPredictions, hidden, output);
  predictions.clear();
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
    predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
  std::vector<std::pair<real,std::string>> predictions;
  while (in.peek() != EOF) {
    predict(in, k, predictions);
    if (predictions.empty()) {
      std::cout << "n/a" << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << ' ';
      }
      std::cout << it->second;
      if (print_prob) {
        std::cout << ' ' << exp(it->first);
      }
    }
    std::cout << std::endl;
  }
}

void FastText::wordVectors() {
  std::string word;
  Vector vec(args_->dim);
  while (std::cin >> word) {
    getVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
}

void FastText::textVectors() {
  std::vector<int32_t> line, labels;
  Vector vec(args_->dim);
  while (std::cin.peek() != EOF) {
    dict_->getLine(std::cin, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    vec.zero();
    for (auto it = line.cbegin(); it != line.cend(); ++it) {
      vec.addRow(*input_, *it);
    }
    if (!line.empty()) {
      vec.mul(1.0 / line.size());
    }
    std::cout << vec << std::endl;
  }
}

void FastText::printVectors() {
  if (args_->model == model_name::sup) {
    textVectors();
  } else {
    wordVectors();
  }
}

void FastText::trainThread(int32_t threadId) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, threadId);
  
  if (args_->model == model_name::sup) {
    model.setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model.setTargetCounts(dict_->getCounts(entry_type::word));
  }

  const int64_t ntokens = dict_->ntokens();
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  while (tokenCount < args_->epoch * ntokens) {
    real progress = real(tokenCount) / (args_->epoch * ntokens);
    real lr = args_->lr * (1.0 - progress);
    localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
    if (args_->model == model_name::sup) {
      dict_->addNgrams(line, args_->wordNgrams);
      supervised(model, lr, line, labels);
    } else if (args_->model == model_name::cbow) {
      cbow(model, lr, line);
    } else if (args_->model == model_name::sg) {
      skipgram(model, lr, line);
    }
    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount += localTokenCount;
      localTokenCount = 0;
      if (threadId == 0 && args_->verbose > 1) {
        printInfo(progress, model.getLoss());
      }
    }
  }
  if (threadId == 0 && args_->verbose > 0) {
    printInfo(1.0, model.getLoss());
    std::cout << std::endl;
  }
  ifs.close();
}

 void FastText::reduceCorpus(int32_t threadId) {
      std::ofstream ofs(args_->input + ".reduced.txt");
      if (!ofs.is_open()) {
        throw std::invalid_argument(
                args_->output + ".reduced.txt" + " cannot be opened for saving vectors!");
      }

      std::ifstream ifs(args_->input);
      utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

      Model model(input_, output_, args_, threadId);

      const int64_t ntokens = dict_->ntokens();
      int64_t localTokenCount = 0;
      std::vector<std::string> line, labels;
      std::cout << "Full context: " << ntokens << std::endl;
      int32_t reduced_context = static_cast<int32_t>( ntokens / 12);
      while (tokenCount < reduced_context) {
        localTokenCount += dict_->getLine(ifs, line, model.rng);
        tokenCount += localTokenCount;
        localTokenCount = 0;
        for(int32_t t =0; t < line.size();t++ ) {
          ofs << line[t] << " ";
        }
        ofs << std::endl;

      }
      std::cout << "Reduced context: " << tokenCount << std::endl;
      ifs.close();
    } 


void FastText::loadVectors(std::string filename) {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  in >> n >> dim;
  std::cout << "Types:" << n << "dim:" << dim;
  if (dim != args_->dim) {
    std::cerr << "Dimension of pretrained vectors does not match -dim option"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    //dict_->add("",word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->data_[i * dim + j];
    }
  }
  in.close();

  //dict_->threshold(1);
  input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getHash(words[i]);
    if (idx < 0 || idx >= dict_->nwords() + args_->bucket) continue;
    for (size_t j = 0; j < dim; j++) {
      input_->data_[idx * dim + j] = mat->data_[i * dim + j];
    }
  }
}

void FastText::train(std::shared_ptr<Args> args) {
  args_ = args;
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    std::cerr << "Cannot use stdin for training!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    std::cerr << "Input file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  dict_->readFromFile(ifs);
  ifs.close();

  if (args_->pretrainedVectors.size() != 0) {
    std::cout << "loading pre-trained subwords from: " << args_->pretrainedVectors << std::endl;
    loadVectors(args_->pretrainedVectors);
  } else {
    input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
    input_->uniform(1.0 / args_->dim);
  }

  if (args_->model == model_name::sup) {
    output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
  } else {
    output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
  }
  output_->zero();

//  reduceCorpus(0);
//  exit(-1);

  start = clock();
  tokenCount = 0;
  std::vector<std::thread> threads;
  std::cout <<  "Created threads: " << dict_->nwords() << std::endl;
  for (int32_t i = 0; i < args_->thread; i++) {
    threads.push_back(std::thread([=]() { trainThread(i); }));
  }
  for (auto it = threads.begin(); it != threads.end(); ++it) {
    it->join();
  }
  model_ = std::make_shared<Model>(input_, output_, args_, 0);

  //saveModel();
  if (args_->model != model_name::sup) {
    saveVectors();
  }
}

}
