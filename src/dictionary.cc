/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <assert.h>
#include <sstream> 
#include <iostream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <cctype>

#include "utils.h"

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
static const char PROP_VALUE_SEP = ':';
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";


Dictionary::Dictionary(std::shared_ptr<Args> args) {
  args_ = args;
  size_ = 0;
  lsize_ = 0;
  msize_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  ntokens_ = 0;
  word2int_.resize(MAX_VOCAB_SIZE);
  l2int_.resize(MAX_VOCAB_SIZE);
  m2int_.resize(MAX_VOCAB_SIZE);
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
    l2int_[i] = -1;
    m2int_[i] = -1;
  }
}

int32_t Dictionary::find(const std::string& w) const {
  int32_t h = hash(w) % MAX_VOCAB_SIZE;
  while (word2int_[h] != -1 && words_[word2int_[h]].word != w) {
    h = (h + 1) % MAX_VOCAB_SIZE;
  }
  return h;
}


void Dictionary::add(const std::string& w, const std::string& aw) {
  int32_t h = find(w);
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = (w.find(args_->label) == 0) ? entry_type::label : entry_type::word;
    e.actual_word = aw;
    words_.push_back(e);
    word2int_[h] = size_++;
  } else {
    words_[word2int_[h]].count++;
  }
}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

const std::vector<int32_t>& Dictionary::getNgrams(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<subentry>& Dictionary::getSubNgrams(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subngram;
}

const std::vector<subentry>& Dictionary::getSubIpaNgrams(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subipangram;
}

const subentry& Dictionary::getLemma(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].sublemma;
}

const std::vector<subentry>& Dictionary::getMorph(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].submorph;
}

const std::vector<int32_t> Dictionary::getNgrams(const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getNgrams(i);
  }
  std::vector<int32_t> ngrams;
  std::vector<subentry> subngram;
  std::vector<subentry> subipangram;
  subentry sublemma;
  std::vector<subentry> submorph;
  computeNgrams(word, ngrams,subngram,sublemma,submorph,subipangram);
  return ngrams;
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

int32_t Dictionary::getHash(const std::string& str) const{
	int32_t h = hash(str) % args_->bucket;
	return nwords_ + h;
}

uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

void Dictionary::computeNgrams(const std::string& word,
                               std::vector<int32_t>& ngrams,
			      std::vector<subentry>& subngram,
	                      subentry& sublemma,
                              std::vector<subentry>& submorph,
			      std::vector<subentry>& subipangram) const {
  //char x;
  std::vector<std::string> propsValues = utils::split(word, '~');
  for (size_t i = 0; i < propsValues.size(); i++) {
    std::string value = propsValues[i];
	std::string prop = value.substr(0, value.find(PROP_VALUE_SEP));
  const bool useProp = args_->props.find(prop) != args_->props.end();
	if (useProp) {
	   if(prop == "w"){

        // If only whole word  uncomment next two lines and comment rest code in the if loop
		/*int32_t h = hash(value) % args_->bucket;
		ngrams.push_back(nwords_ + h);*/

       // Default consider character-ngrams
	   std::string temp_word = BOW + value.substr(value.find(PROP_VALUE_SEP)+1,std::string::npos) + EOW;
	   for (size_t i = 0; i < temp_word.size(); i++) {
            std::string charn;
            if ((temp_word[i] & 0xC0) == 0x80) continue;
            for (size_t j = i, n = 1; j < temp_word.size() && n <= args_->maxn; n++) {
              charn.push_back(temp_word[j++]);
              while (j < temp_word.size() && (temp_word[j] & 0xC0) == 0x80) {
                charn.push_back(temp_word[j++]);
              }
              if (n >= args_->minn && !(n == 1 && (i == 0 || j == temp_word.size()))) {
                int32_t h = hash(charn) % args_->bucket;
                ngrams.push_back(nwords_ + h);
		        subentry e;
		        e.subword = charn;
		        e.id = nwords_+h;
		        subngram.push_back(e);
              }
            }
          }
	}
	else{
         std::string val = value.substr(value.find(PROP_VALUE_SEP)+1, std::string::npos);
         if(prop == "l"){
	  	    int32_t h = hash(value) % args_->bucket;
            sublemma.id = nwords_+h;
		    sublemma.subword = value;
	        ngrams.push_back(nwords_ + h);
	      }
	    if(prop == "m"){
            //Default encodes each morph tag separately
		    std::vector<std::string> morphValues = utils::split(val, '+');

		    for(int32_t m=0;m<morphValues.size();m++){
			    int32_t h = hash( morphValues[m]) % args_->bucket;
			    subentry e;
			    e.id = nwords_+h;
			    e.subword =  morphValues[m];
			    submorph.push_back(e);
	   		    ngrams.push_back(nwords_ + h);
		    }

          //Encodes the morphology tags as a whole
		  /*subentry e;
		  std::string val = value.substr(value.find(PROP_VALUE_SEP)+1,std::string::npos);
		  if(value=="")
		  { value = "m:<unk>";}
		  int32_t h = hash(value) % args_->bucket;
		  e.id = nwords_+h;
		  e.subword = value;
		  submorph.push_back(e);
		  ngrams.push_back(nwords_+h);*/
	    }

        if(prop == "ipa"){
	
	      std::string temp_word = BOW + value.substr(value.find(PROP_VALUE_SEP)+1,std::string::npos) + EOW;
	      for (size_t i = 0; i < temp_word.size(); i++) {
                std::string charn;
                if ((temp_word[i] & 0xC0) == 0x80) continue;
                for (size_t j = i, n = 1; j < temp_word.size() && n <= args_->maxn; n++) {
                    charn.push_back(temp_word[j++]);
                    while (j < temp_word.size() && (temp_word[j] & 0xC0) == 0x80) {
                    charn.push_back(temp_word[j++]);
                 }
              if (n >= args_->minn && !(n == 1 && (i == 0 || j == temp_word.size()))) {
                int32_t h = hash(charn) % args_->bucket;
                ngrams.push_back(nwords_ + h);
		        subentry e;
		        e.subword = charn;
		        e.id = nwords_+h;
		        subipangram.push_back(e);
              }
            }
          }
	    }
	  }
	}
  }
}

void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
  	 words_[i].subwords.clear();
 	 words_[i].subwords.push_back(i);
	 subentry sl;
	 subentry sm;
         words_[i].sublemma = sl;
   	 computeNgrams(words_[i].actual_word, words_[i].subwords, words_[i].subngram,words_[i].sublemma,words_[i].submorph,words_[i].subipangram);
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const
{
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  while (readWord(in, word)) {
    if (word == EOS)
      continue;
    
    
    std::vector<std::string> propsValues = utils::split(word, '~');
    std::string value = propsValues[0];
    std::string only_word = value.substr(value.find(PROP_VALUE_SEP)+1, std::string::npos);
    
    add(only_word,word);

    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cout << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
    }
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      threshold(minThreshold++);
    }
  }
  threshold(args_->minCount);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cout << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
    std::cout << "Number of words:  " << nwords_ << std::endl;
    std::cout << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    std::cerr << "Empty vocabulary. Try a smaller -minCount value." << std::endl;
    exit(EXIT_FAILURE);
  }
}

void Dictionary::threshold(int64_t t) {
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.count > e2.count;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
        return e.type == entry_type::word && e.count < t;
      }), words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) counts.push_back(w.count);
  }
  return counts;
}

void Dictionary::addNgrams(std::vector<int32_t>& line, int32_t n) const {
  int32_t line_size = line.size();
  for (int32_t i = 0; i < line_size; i++) {
    uint64_t h = line[i];
    for (int32_t j = i + 1; j < line_size && j < i + n; j++) {
      h = h * 116049371 + line[j];
      line.push_back(nwords_ + (h % args_->bucket));
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(std::istream& in,
                                std::vector<std::string>& words,
                                std::minstd_rand& rng) const {
      std::uniform_real_distribution<> uniform(0, 1);
      std::string token;
      int32_t ntokens = 0;

      reset(in);
      words.clear();
      while (readWord(in, token)) {
        int32_t h = find(token);
        int32_t wid = word2int_[h];
        if (wid < 0) continue;

        ntokens++;
        if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
          words.push_back(token);
        }
        if (ntokens > MAX_LINE_SIZE || token == EOS) break;
      }
      return ntokens;
    }



int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::vector<int32_t>& labels,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;
  words.clear();
  labels.clear();
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
  while (readWord(in, token)) {
    if (token == EOS) break;

    //SplitWord
    std::vector<std::string> propsValues = utils::split(token, '~');
    std::string value = propsValues[0];
    //std::cout << value << std::endl;
    std::string word = value.substr(value.find(PROP_VALUE_SEP)+1, std::string::npos);

    int32_t wid = getId(word);
    if (wid < 0) continue;
    entry_type type = getType(wid);
    ntokens++;
    if (type == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (type == entry_type::label) {
      labels.push_back(wid - nwords_);
    }
    if (words.size() > MAX_LINE_SIZE && args_->model != model_name::sup) break;
  }
  return ntokens;
}

std::string Dictionary::getLabel(int32_t lid) const {
  assert(lid >= 0);
  assert(lid < nlabels_);
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*) &size_, sizeof(int32_t));
  out.write((char*) &nwords_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*) &(e.count), sizeof(int64_t));
    out.write((char*) &(e.type), sizeof(entry_type));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  for (int32_t i = 0; i < MAX_VOCAB_SIZE; i++) {
    word2int_[i] = -1;
  }
  in.read((char*) &size_, sizeof(int32_t));
  in.read((char*) &nwords_, sizeof(int32_t));
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*) &e.count, sizeof(int64_t));
    in.read((char*) &e.type, sizeof(entry_type));
    words_.push_back(e);
    word2int_[find(e.word)] = i;
  }
  initTableDiscard();
  initNgrams();
}

}
