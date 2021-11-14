# Wolof Translation 
 
<p align="center">
<img src="./input/aa.jpg"  width="312" height="297">
</p>


## Description


Machine translation is an area of NLP research that allows one language to be translated into another language. Today it is difficult to find an automatic translator for languages that are not well represented, especially in Africa. This is the purpose of this project.

Wolof is a language spoken in Senegal and many other countries in West Africa. However, having a translation tool would make it possible to popularise works written in Wolof, but also to contribute to the representation of the language's resources.



## Objective

To achieve our objects in the project we will implement two models based on the architecture of transformers, namely :

* A model based on seq2seq transformers for more information I invite you to visit the [site](https://huggingface.co/transformers/model_doc/encoderdecoder.html)

* A model based on [T5](https://huggingface.co/transformers/model_doc/t5.html)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

 
install the dependencies for this project by running the following commands in your terminal:

```
 pip install -r requirements.txt
```

run the seq2seq model by running the following command in your terminal:

```
python src/seq2seq.py --train_file="./input/wolof.csv" \
                        --max_source_length=150 \
                        --max_target_length=150 \
                        --number_epochs=4 \
                        --learning_rate=3e-8 \
                        --epsilone=1e-9 \
                        --train_batch_size=3 \
                        --model_name="bert-base-cased"
```

run the t5 model by running the following command in your terminal:

```
python src/t5.py --train_file="./input/wolof.csv" \
                        --max_source_length=150 \
                        --max_target_length=150 \
                        --number_epochs=4 \
                        --learning_rate=3e-8 \
                        --epsilone=1e-9 \
                        --train_batch_size=3 \
                        --model_name="t5-small" \
                        --task_prefix="translation French to Wolof: "
```






## Ressources 

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

[Transformers-based Encoder-Decoder Models](https://huggingface.co/blog/encoder-decoder#encoder-decoder)

[A Review of the Neural History of Natural Language Processing](https://ruder.io/a-review-of-the-recent-history-of-nlp)

[Neurail Machine Translation By Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)



 










