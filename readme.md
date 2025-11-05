# Modular Memory-augmented Transformer (MoMeT)
## Description
This project is a proof-of-concept for a modular architecture that orchestrates multiple Transformer models, each specialized for a narrowly-scoped task i.e [Named-Entity Recognition](https://www.ibm.com/think/topics/named-entity-recognition), [summarization](https://www.ibm.com/think/topics/text-summarization) and [text generation](https://www.ibm.com/think/topics/text-generation). Its primary purpose is to to demonstrate, in practice, some of the concepts outlined in docs/Technical Design Document (Refined).pdf (TL;DR The document discusses a Machine Learning architectural design that combines the following technologies: [RAG](https://doi.org/10.48550/arXiv.2405.06211) + [Memory-Augmented Neural Network](https://doi.org/10.48550/arXiv.2312.06141)  + [Information Retrieval System](https://ciir.cs.umass.edu/irbook/). The aim of the design is to fix some of the challenges that Machine Learning models specifically Transformer models face. The system involves a combination of multiple modular components: Transformer models + Information Retrieval subsystem, where each component explicitly focuses on a singular task adhering strictly to the [separation-of-concerns principles](https://nalexn.github.io/separation-of-concerns/).)

## How it works
The project relies on a synthetically generated dataset, constructed from a collection of sentences templates that are randomly selected and populated with variable inputs such as people names. Each generated record is structured as follows:
* Content: This contains a variety of details about a person separated with a semicolon (;). Details include the individual's names including their mother and father's name, their occupation, where they live, likes, dislikes, hobbies, and the university they attended.
* Context: This contains a grouping of prompts and responses grouped by the details found in the Content.
    * Prompt: Specifies the exact query to be provided to the model.
    * Response: This includes variations of response to a given prompt.

The system incorporates three specialized models:
* Model_0: This is a Deconder-only Transformer model used for Named-Entity Recognition where given a prompt, it will output the person's name found in the prompt text.
* Model_1: This is an Encoder-Decoder Transformer model that takes the entire content data and summarizes the relevant information from it using based on the prompt given.
* Model_2: This is an Encoder-Decoder Transformer model that generates a coherent and accurate response to the prompt based on the summarization data.

<p align="center">
  <img alt="Visual depiction of MoMeT" src="assets/MoMeT%20Diagram.jpg" />
</p>

## Implementation
The project includes the following functionalities:
* Scripts to generate the synthetic dataset and vocabularies, tokenize and save them to file.
* Code to train all the models (Model_0, Model_1, Model_2).
* Code to test models capabilities using pre-trained models and testing dataset (Dataset held out from training process).

## Requirements
* Anaconda (Optional)
* Python 3
* Faker library

## Installing.
1. (Optional) Install [Anaconda](https://docs.anaconda.com/) on your machine.
    * Create anaconda environment:
    ```
    conda create --name <env_name> python=3.12
    ```
    * To activate anaconda environment, if not already activated:
    ```
    conda activate <env_name>
    ```
2. (If not installed) Install [Pytorch 2.5](https://pytorch.org/get-started/locally/) based on your hardware requirements and preferences.
3. Install Python depencies:
    ```
    pip install -r requirements.txt
    ```

## Generating dataset.
To generate the synthetic dataset:
* Generate JSON text dataset file
```
python Scripts/generate_text_dataset.py --dest-path <Destination output path for dataset json> --lists-path <File path to CSV Lists> --template-path <File path to JSON Template> --num-training-data <Number of training dataset> --num-testing-data <Number of testing dataset>
```

* Generate JSON vocabulary file
```
python Scripts/generate_vocabulary.py --dest-path <Destination output path for dataset json> --lists-path <File path to CSV Lists>
```

* Generate token datasets files
```
python Scripts/generate_token_dataset.py --dest-path <Destination output path for dataset json> --dictionary-path <File path to generated Dictionary> --text-dataset-path <File path to generated text Dataset>
```

## Creating config file.
Before a model can be trained, you will be required to generate a config file (config.json) containing information about the model parameters and other information such as learning rate. It should look like this, adjust accordingly:
```
{
    "model_lr": 1e-4,
    "num_heads": 32,
    "embedding_dim": 512,
    "hidden_dim": 2048,
    "context_window": 256,
    "activation_type": "gelu",
    "max_global_steps": 100000,
    "lr_gamma": 0.5,
    "num_encoder_blocks": 4,
    "num_decoder_blocks": 4
}
```

## Training the models.
To train the model(s) using the generated dataset and config file, run the following (adjust accordingly):
```
python train_Transformer_models.py --device <Which hardware device will model run on> --temperature <Temperature parameter for softmax sampling> --model-type <Model to be trained>  --vocabulary-path <File path to vocabulary> --tr-dataset-path <File path to training json dataset file> --tst-dataset-path <File path to testing json dataset file> --batch-size <Batch size of dataset> --checkpoint-steps <Steps for checkpointing and/or testing model> --config-path <File path to JSON config file> --out-dir <Folder path of output directory> --model-checkpoint <File path to model checkpoint to load from (if any)> --lr-steps <Global steps in between halving learning rate> --load-optim <Optional flag to load pre-trained model's optimizer>
```

## Testing Pre-trained models.
To test all the pre-trained models on the testing dataset:
```
python test_text_generation_POC.py --device <Which hardware device will model run on> --temperature <Temperature parameter for softmax sampling> --vocabulary-path <File path to vocabulary> --tst-dataset-path <File path to testing json dataset file> --model-0-checkpoint <File path to model_0 model's checkpoint> --model-1-checkpoint <File path to model_1 model's checkpoint> --model-2-checkpoint <File path to model_2 model's checkpoint>
```

## Example
The models were trained on a sinlge **RTX 3080** 10GB GPU. Below is a screenshot of each of the model's response and their input data.

<p align="center">
  <img alt="Text generation example" src="assets/Text generation example.png" />
</p>
