# subword2vec
subword2vec is the code repository for training word embeddings enriched with sub-word knowledge like character ngrams, lemma, morphological tags and phonemes. This library was used in our upcoming paper `Adapting Word Embeddings to New Languages with Morphological and Phonological Subword Representations` (to appear in EMNLP-2018). subword2vec is based on the [fastText](https://github.com/facebookresearch/fastText) library and [prop2vec](https://github.com/oavraham1/prop2vec) library. 

### Requirements
- gcc-4.6.3 or newer (for compiling)

### Input format
Each word is represented with its sub-word information. For instance, a Hindi word `नदी `(river) is represented as `w:नदी~l:नद~m:N+Fem+Dir+Sg~ipa:nədiː`. An example input file is shown in the `\example` directory.

### Command
```
cd Embeddings
make
./fasttext skipgram -input example/sample_input.txt -output example/sample_output -lr 0.025 -dim 100 -t 1e-3 -props w+l+m -minCount 2 -ws 3 -bucket 2000000 -lemmaoutput example/sample_lemma_output -morphoutput example/sample_morph_output
```

This command will train embeddings by averaging the sub_word units: charngrams (w:), lemma (l:), morph(m:). One can provide different combinations in the `-props` field depending on one's requirements. If one needs a combination of `charngrams + morph`, one should provide `-props w+m`. The arguments `-lemmaoutput` and `-morphoutput` would output the embeddings of the each sub-word unit. 

##### Using pretrained sub-word embeddings
To use pretrained embeddings for initializing vectors for the sub-words, the pretrained embeddings should have the following format:
```
102345 100
Punc 1.4163 1.697 -0.95646 0.4587 1.2924 0 ...
```
where `102345` denotes the number of unique sub-words `100` denotes the embedding size. Then run the following:
```
./fasttext skipgram -input example/sample_input.txt -output  example/sample_output_pretrained_with_morph -ws 3 -t 1e-3 -minCount 2 -lr 0.025 -bucket 2000000 -props w+l+m -pretrainedVectors example/morph_output.vec
```
### Best embeddings (in reference to the paper)
The embeddings which gave best performance in the NER task used for our work here `Adapting Word Embeddings to New Languages with Morphological and Phonological Subword Representations` (to appear in EMNLP-2018), are made available in this folder: `\embeddings_released`. 
