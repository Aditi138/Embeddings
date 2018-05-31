/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "utils.h"

#include <cmath>
#include <ios>

namespace fasttext {

namespace utils {
  real* t_sigmoid = nullptr;
  real* t_log = nullptr;

  real log(real x) {
    if (x > 1.0) {
      return 0.0;
    }
    int i = int(x * LOG_TABLE_SIZE);
    return t_log[i];
  }

  real sigmoid(real x) {
    if (x < -MAX_SIGMOID) {
      return 0.0;
    } else if (x > MAX_SIGMOID) {
      return 1.0;
    } else {
      int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
      return t_sigmoid[i];
    }
  }

  void initTables() {
    initSigmoid();
    initLog();
  }

  void initSigmoid() {
    if (t_sigmoid != nullptr) return;
    t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
      real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
      t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
    }
  }

  void initLog() {
    if (t_log != nullptr) return;
    t_log = new real[LOG_TABLE_SIZE + 1];
    for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
      real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
      t_log[i] = std::log(x);
    }
  }

  void freeTables() {
    delete[] t_sigmoid;
    delete[] t_log;
    t_sigmoid = nullptr;
    t_log = nullptr;
  }

  int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }
	
  const std::vector<std::string> split(const std::string &text, char sep) {
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;
    while ((end = text.find(sep, start)) != std::string::npos) {
      tokens.push_back(text.substr(start, end - start));
      start = end + 1;
    }
    tokens.push_back(text.substr(start));
    return tokens;
  }
}

}
