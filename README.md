# AllenNLP-Models
The goal is to experiment using pretrained models in [AllenNLP](https://allennlp.org/) to make predictions on sentences labeled as gender-biased from [biaslyAI](biaslyAI.com). 

This script takes sample sentences from `biased_sentences.json` and uses AllenNLP's pretrained models (Semantic role labeling or Co-reference Resolution) to make predictions. It outputs the predictions in `biased_sentences_srl.json` or `biased_sentences_coref.json` respectively.

## Install AllenNLP

I installed AllenNLP using pip3:

```
pip3 install allennlp
```

But there are other options: https://github.com/allenai/allennlp#installation

## Run Script on Pretrained AllenNLP Models

### Semantic Role Labeling 
Semantic Role Labeling (SRL) recovers the latent predicate argument structure of a sentence, providing representations that answer basic questions about sentence meaning, including “who” did “what” to “whom,” etc. The AllenNLP SRL model is a reimplementation of a deep BiLSTM model [(He et al, 2017)](https://www.semanticscholar.org/paper/Deep-Semantic-Role-Labeling%3A-What-Works-and-What's-He-Lee/a4dd3beea286a20c4e4f66436875932d597190bc), which is currently state of the art for PropBank SRL (Newswire sentences). [Source](https://allennlp.org/)

To run script with SRL model

```
python3 allennlp_models.py \
    https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.02.27.tar.gz \
    biased_sentences.json --output-file biased_sentences_srl.json
```

### Co-reference Resolution
Coreference resolution is the task of finding all expressions that refer to the same entity in a text. End-to-end Neural Coreference Resolution [(Lee et al, 2017)](https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/8ae1af4a424f5e464d46903bc3d18fe1cf1434ff) is a neural model which considers all possible spans in the document as potential mentions and learns distributions over possible anteceedents for each span, using aggressive, learnt pruning strategies to retain computational efficiency. It achieved state-of-the-art accuracies on the [Ontonotes 5.0 dataset](http://cemantix.org/data/ontonotes.html) in early 2017. [Source](https://allennlp.org/)


To run script with Co-reference model

```
python3 allennlp_models.py \
    https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz \
    biased_sentences.json --output-file biased_sentences_coref.json
```

## Visualization

To better visualize the output predictions, you can run a Flask server that will serve predictions from a single AllenNLP model.
It also includes a very, very bare-bones web front-end for exploring predictions.

To visualize the output for Semantic Role Labeling model, run the below command and navigate to `localhost:8000`

```
python3 -m allennlp.service.server_simple \
    --archive-path https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.02.27.tar.gz \
    --predictor semantic-role-labeling \
    --title "AllenNLP Semantic Role Labeling on biaslyAI Sentences" \
    --field-name sentence
```

To visualize the output for Co-reference Resolution model, run the below command and navigate to `localhost:8000`

```
python3 -m allennlp.service.server_simple \
    --archive-path https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz \
    --predictor coreference-resolution \
    --title "AllenNLP Co-reference on biaslyAI Sentences" \
    --field-name document
```

For a much better front-end visual of the models, check out AllenNLP's demos:

Semantic Role Labeling: http://demo.allennlp.org/semantic-role-labeling

Co-reference Resolution: http://demo.allennlp.org/coreference-resolution
