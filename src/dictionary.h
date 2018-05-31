/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_DICTIONARY_H
#define FASTTEXT_DICTIONARY_H

#include <vector>
#include <string>
#include <istream>
#include <ostream>
#include <random>
#include <memory>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t {word=0, label=1};
struct subentry {
    std::string subword;
    int64_t id;
 };

struct entry {
  std::string word;
  int64_t count;
  entry_type type;
  std::vector<int32_t> subwords;
  std::vector<subentry> subngram;
  std::vector<subentry> subipangram;
  subentry sublemma;
  std::vector<subentry> submorph;
  std::string actual_word;
};

class Dictionary {
  private:
    static const int32_t MAX_VOCAB_SIZE = 30000000;
    static const int32_t MAX_LINE_SIZE = 1024;

    int32_t find(const std::string&) const;
    void initTableDiscard();
    void initNgrams();

    std::shared_ptr<Args> args_;
    std::vector<int32_t> word2int_;
    std::vector<int32_t> l2int_;
    std::vector<int32_t> m2int_;
    std::vector<entry> words_;
    std::vector<real> pdiscard_;
    int32_t size_;
    int32_t lsize_;
    int32_t msize_;
    int32_t nwords_;
    int32_t nlabels_;
    int64_t ntokens_;

  public:
    static const std::string EOS;
    static const std::string BOW;
    static const std::string EOW;

    explicit Dictionary(std::shared_ptr<Args>);
    int32_t nwords() const;
    int32_t nlabels() const;
    int64_t ntokens() const;
    int32_t getId(const std::string&) const;
    entry_type getType(int32_t) const;
    bool discard(int32_t, real) const;
    std::string getWord(int32_t) const;
    const std::vector<int32_t>& getNgrams(int32_t) const;
    const subentry& getLemma(int32_t) const;
    const std::vector<subentry>& getMorph(int32_t) const;
    const std::vector<subentry>& getSubNgrams(int32_t) const;
    const std::vector<subentry>& getSubIpaNgrams(int32_t) const;
    const std::vector<int32_t> getNgrams(const std::string&) const;
    void computeNgrams(const std::string&, std::vector<int32_t>&i, std::vector<subentry>&, subentry&, std::vector<subentry>&,std::vector<subentry>&) const ;
    uint32_t hash(const std::string& str) const;
    void add(const std::string&, const std::string&);
    bool readWord(std::istream&, std::string&) const;
    void readFromFile(std::istream&);
    int32_t getHash(const std::string& str) const;
    std::string getLabel(int32_t) const;
    void save(std::ostream&) const;
    void load(std::istream&);
    std::vector<int64_t> getCounts(entry_type) const;
    void addNgrams(std::vector<int32_t>&, int32_t) const;
    int32_t getLine(std::istream&, std::vector<int32_t>&,
                    std::vector<int32_t>&, std::minstd_rand&) const;
    int32_t getLine(std::istream&, 
                    std::vector<std::string>&, std::minstd_rand&) const;
    void threshold(int64_t);
    void reset(std::istream&) const;
};

}

#endif
