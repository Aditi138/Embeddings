# subword2vec
subword2vec is the code repository for training word embeddings enriched with sub-word knowledge like character ngrams, lemma, morphological tags and phonemes. This library was used in the paper `Adapting Word Embeddings to New Languages with Morphological and Phonological Subword Representations` (to appear in EMNLP-2018). subword2vec is based on the [fastText](https://github.com/facebookresearch/fastText) library and [prop2vec](https://github.com/oavraham1/prop2vec) library. 

### Requirements
- gcc-4.6.3 or newer (for compiling)

### Input format
Each word is represented with its sub-word information. For instance, a Hindi word `नदी `(river) is represented as `w:नदी~l:नद~m:N+Fem+Dir+Sg~ipa:nədiː`. An example input file is shown in the `\example` directory.

### Command
cd 

The word embedding is then represented as the average of the requested sub-word embeddings. 
 The input file should have word representat
