/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "args.h"

#include <stdlib.h>
#include <string.h>

#include <iostream>

#include "utils.h"

namespace fasttext {

static const char PROPS_SEP='+';

Args::Args() {
  lr = 0.05;
  dim = 100;
  ws = 5;
  epoch = 5;
  minCount = 5;
  neg = 5;
  wordNgrams = 1;
  loss = loss_name::ns;
  model = model_name::sg;
  bucket = 2000000;
  thread = 12;
  lrUpdateRate = 100;
  t = 1e-4;
  label = "__label__";
  verbose = 2;
  minn = 3;
  maxn = 6;
  pretrainedVectors = "";
}
	
void Args::initProps(std::string propsStr) {
  props.clear();
  std::vector<std::string> propsVec = utils::split(propsStr, PROPS_SEP);
  for (size_t i = 0; i < propsVec.size(); i++) {
    props.insert(propsVec[i]);
  }
}

void Args::parseArgs(int argc, char** argv) {
  std::string command(argv[1]);
  if (command == "supervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    minCount = 1;
    lr = 0.1;
  } else if (command == "cbow") {
    model = model_name::cbow;
  }
  int ai = 2;
  while (ai < argc) {
    if (argv[ai][0] != '-') {
      std::cout << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    if (strcmp(argv[ai], "-h") == 0) {
      std::cout << "Here is the help! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    } else if (strcmp(argv[ai], "-input") == 0) {
      input = std::string(argv[ai + 1]); 
	} else if (strcmp(argv[ai], "-test") == 0) {
      test = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-output") == 0) {
      output = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-suboutput") == 0) {
      suboutput = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lemmaoutput") == 0) {
      lemmaoutput = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-morphoutput") == 0) {
      morphoutput = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-ipaoutput") == 0) {
      ipaoutput = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-props") == 0) {
      propsStr = std::string(argv[ai + 1]);
      initProps(propsStr);
    } else if (strcmp(argv[ai], "-lr") == 0) {
      lr = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lrUpdateRate") == 0) {
      lrUpdateRate = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dim") == 0) {
      dim = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-ws") == 0) {
      ws = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-epoch") == 0) {
      epoch = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minCount") == 0) {
      minCount = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-neg") == 0) {
      neg = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minn") == 0) {
          minn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai],"-maxn") == 0) {
          maxn = atoi(argv[ai + 1]);
    }else if (strcmp(argv[ai], "-wordNgrams") == 0) {
      wordNgrams = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-loss") == 0) {
      if (strcmp(argv[ai + 1], "hs") == 0) {
        loss = loss_name::hs;
      } else if (strcmp(argv[ai + 1], "ns") == 0) {
        loss = loss_name::ns;
      } else if (strcmp(argv[ai + 1], "softmax") == 0) {
        loss = loss_name::softmax;
      } else {
        std::cout << "Unknown loss: " << argv[ai + 1] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } else if (strcmp(argv[ai], "-bucket") == 0) {
      bucket = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-thread") == 0) {
      thread = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-t") == 0) {
      t = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-label") == 0) {
      label = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-verbose") == 0) {
      verbose = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-pretrainedVectors") == 0) {
      pretrainedVectors = std::string(argv[ai + 1]);
    } else {
      std::cout << "Unknown argument: " << argv[ai] << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    ai += 2;
  }
  if (input.empty() || output.empty()) {
    std::cout << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
}

void Args::printHelp() {
  std::string lname = "ns";
  if (loss == loss_name::hs) lname = "hs";
  if (loss == loss_name::softmax) lname = "softmax";
  std::cout
    << "\n"
    << "The following arguments are mandatory:\n"
    << "  -input              training file path\n"
    << "  -output             output file path\n"
    << "  -props              properties to use (with '+' as separator, e.g. 'w+l+m')\n\n"
    << "The following arguments are optional:\n"
    << "  -lr                 learning rate [" << lr << "]\n"
    << "  -lrUpdateRate       change the rate of updates for the learning rate [" << lrUpdateRate << "]\n"
    << "  -dim                size of word vectors [" << dim << "]\n"
    << "  -ws                 size of the context window [" << ws << "]\n"
    << "  -epoch              number of epochs [" << epoch << "]\n"
    << "  -minCount           minimal number of word occurences [" << minCount << "]\n"
    << "  -neg                number of negatives sampled [" << neg << "]\n"
    << "  -wordNgrams         max length of word ngram [" << wordNgrams << "]\n"
    << "  -loss               loss function {ns, hs, softmax} [ns]\n"
    << "  -bucket             number of buckets [" << bucket << "]\n"
    << "  -thread             number of threads [" << thread << "]\n"
    << "  -t                  sampling threshold [" << t << "]\n"
    << "  -label              labels prefix [" << label << "]\n"
    << "  -verbose            verbosity level [" << verbose << "]\n"
    << "  -pretrainedVectors  pretrained word vectors for supervised learning []"
    << std::endl;
}

void Args::save(std::ostream& out) {
  out.write((char*) &(dim), sizeof(int));
  out.write((char*) &(ws), sizeof(int));
  out.write((char*) &(epoch), sizeof(int));
  out.write((char*) &(minCount), sizeof(int));
  out.write((char*) &(neg), sizeof(int));
  out.write((char*) &(wordNgrams), sizeof(int));
  out.write((char*) &(loss), sizeof(loss_name));
  out.write((char*) &(model), sizeof(model_name));
  out.write((char*) &(bucket), sizeof(int));
  out.write((char*) &(lrUpdateRate), sizeof(int));
  out.write((char*) &(t), sizeof(double));
  
  int len = propsStr.size();
  out.write((char*) &len, sizeof(int));
  out.write(propsStr.c_str(), len);
}

void Args::load(std::istream& in) {
  in.read((char*) &(dim), sizeof(int));
  in.read((char*) &(ws), sizeof(int));
  in.read((char*) &(epoch), sizeof(int));
  in.read((char*) &(minCount), sizeof(int));
  in.read((char*) &(neg), sizeof(int));
  in.read((char*) &(wordNgrams), sizeof(int));
  in.read((char*) &(loss), sizeof(loss_name));
  in.read((char*) &(model), sizeof(model_name));
  in.read((char*) &(bucket), sizeof(int));
  in.read((char*) &(lrUpdateRate), sizeof(int));
  in.read((char*) &(t), sizeof(double));
  
  int len;
  in.read((char*) &len, sizeof(int));
  char* temp = new char[len+1];
  in.read(temp, len);
  temp[len] = '\0';
  propsStr = temp;
  delete [] temp;
  initProps(propsStr);
}

}
