# Modular Memory-augmented Models (MoMeM)
## Description
This project demonstrates, in practice, the concepts outlined in the (WIP) Techinical Design Document: <a href="/docs/(WIP) Technical Design Document v4.pdf" type="application/pdf">read more</a> (PDF can be found here: <b>/docs/(WIP) Technical Design Document v4.pdf</b>). It is a <strong>proof of concept (POC)</strong> for a modular system comprising multiple Transformer models and an Information Retrieval (IR) system, that acts as a form of memory (Not explicityly implemented in the code). Each of these component specializes in a narrowly-scoped task:
* [Named-Entity Recognition](https://www.ibm.com/think/topics/named-entity-recognition)
* [Summarization](https://www.ibm.com/think/topics/text-summarization)
* [Text Generation](https://www.ibm.com/think/topics/text-generation).

The principle idea for this project is to combine various Machine Learning (ML) concepts together to form a modular system that borrows heavily from software design principles:
* [Transformer-based Models](https://doi.org/10.48550/arXiv.1706.03762)
* [Memory-Augmented Neural Networks (MANN)](https://doi.org/10.48550/arXiv.2312.06141)
* [Retrieval Augmented Generation (RAG)](https://doi.org/10.48550/arXiv.2405.06211)
* [Case-Based Reasoning (CBR)](https://doi.org/10.1007/BF00155578)
* [Information Retrieval (IR) system](https://ciir.cs.umass.edu/irbook/)

## How it works
The project relies on a synthetically generated toy dataset, constructed from a collection of sentences templates that are randomly selected and populated with variable inputs. Each generated record is structured as follows:
* <b>Content</b>: This contains a variety of details about a fictional person separated with a semicolon (;). Details of the person include the individual's names, their occupation, where they live, favourite movies, favourite music, hobbies, and the university they attended.
* <b>Context</b>: This contains:
    * <b>Prompt</b>: Specifies the exact query to emulate a possible user input request.
    * <b>Response</b>: This includes numerous potential responses to a given prompt.
    * <b>Summary</b>: This contains a text snippet from the <b>Content</b> relevant to the prompt.

Theoretically the system incorporates four modules:
* <b>Module<sub>0</sub></b>: This module contains a Decoder-only Transformer-based model. Its sole task is to perform <b>Named-Entity Recognition</b> on a given prompt, whereby given a prompt, it will output the person's name found in the prompt text as a form of [Tag](https://en.wikipedia.org/wiki/Tag_(metadata)).
* <b>Module<sub>1</sub></b>: This module contains a database (Memory) that stores a Key-Value pairs (<b>Tag</b>: <b>Content</b>). It's main role in the overall system is to store all relevant data in the form of <b>Content</b>, search, and filter them based on the <b>Tag</b> from <b>Module<sub>0</sub></b>. This is not explicitly implemented in the project's code but it's theoretically a key component in the overall system.
* <b>Module<sub>2</sub></b>: This module contains an Encoder-Decoder Transformer-based model. Its sole task is to <b>Summarize</b> the <b>Content</b> provided from <b>Module<sub>1</sub></b>.
* <b>Module<sub>3</sub></b>: This module contains an Encoder-Decoder Transformer-based model. Its sole purpose is to perform <b>Text Generation</b> and produce coherent and accurate response to the prompt based on the <b>Summarized</b> information from <b>Module<sub>2</sub></b>.

## Conceptual overview of the system
<p align="center">
  <img alt="Visual depiction of MoMeM architecture" src="assets/System Diagram (Scaled).jpg" />
</p>

## Implementation
The project includes the following core capabilities:
* Code to generate the synthetic toy dataset and token vocabulary.
* Code to train all the models in their respective modules.
* Code to generate text from saved pre-trained models.

## Requirements
* Anaconda (Optional)
* Python 3

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

## Generating dataset.
To generate the synthetic dataset:
* Generate JSON vocabulary file
```
python Scripts/generate_vocabulary.py --dest-path <Destination output path for dataset json> --lists-path <File path to CSV Lists> --template-path <File path to JSON Template>
```

* Generate JSON text dataset file
```
python Scripts/generate_text_dataset.py --dest-path <Destination output path for dataset json> --lists-path <File path to CSV Lists> --template-path <File path to JSON Template> --context-window <Context window (Character-Based Tokens)> --num-datapoints <Size of dataset in terms of records> --prob-threshold <Probability threshold used to augment dataset with shuffled characters.>
```

* Generate token datasets files
```
python Scripts/generate_token_dataset.py --dest-path <Destination output path for dataset json> --vocabulary-path <File path to generated Vocabulary> --text-dataset-path <File path to generated text Dataset>
```

## Creating config file.
Before a model can be trained, you will be required to generate a config file (**config.json**) containing information about the model parameters and other information such as learning rate. It should look like this, adjust accordingly:
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
    "num_encoder_blocks": 3,
    "num_decoder_blocks": 4
}
```

## Training the models.
To train the model(s) using the generated dataset and config file, run the following (adjust accordingly):
```
python train_Transformer_models.py --device <Which hardware device will model run on> --temperature <Temperature parameter for softmax sampling> --model-type <Model to be trained>  --vocabulary-path <File path to vocabulary> --tr-dataset-path <File path to training json dataset file> --tst-dataset-path <File path to testing json dataset file> --batch-size <Batch size of dataset> --checkpoint-steps <Steps for checkpointing and/or testing model> --config-path <File path to JSON config file> --out-dir <Folder path of output directory> --model-checkpoint <File path to model checkpoint to load from (if any)> --lr-steps <Global steps in between halving learning rate> --load-optim <Optional flag to load pre-trained model's optimizer>
```

## Generating Text from Pre-trained models.
To generate text from pre-trained models run the following, adjust accordingly:
```
python test_text_generation_POC.py --device <Which hardware device will model run on> --temperature <Temperature parameter for softmax sampling> --vocabulary-path <File path to vocabulary> --tst-dataset-path <File path to testing json dataset file> --model-0-checkpoint <File path to model_0 model's checkpoint> --model-1-checkpoint <File path to model_1 model's checkpoint> --model-2-checkpoint <File path to model_2 model's checkpoint>
```

To update the values of the text, edit the following document accordingly: ```test_data.json```

## Hardware
The models were trained on a sinlge **RTX 3080** 10GB GPU. Below is a screenshot of the nvidia-smi output.

<p align="center">
  <img alt="nvidia-smi screenshot" src="assets/nvidia-smi-screenshot.png"/>
</p>

## Pre-trained Models
The pre-trained models can be found [here](https://huggingface.co/VinML/MoMeM-POC) ("All_models.zip" - 2.42GB file). 

## Results
### Best Case (Everything functions properly)
<p align="center">
  <img alt="Text generation example" src="assets/Text Generation (Best Case).png" />
</p>

### Worst Case (Hallucinations and other mistakes)
<p align="center">
  <img alt="Text generation example" src="assets/Text Generation (Worst Case).png" />
</p>

## Observations
The following was observed during training and interacting with the system:
* Due to the small sample size of the toy dataset and model size, the capabilities of the models is very limited. This is observable in the quality of the text generation output where quality degrades when the input variables deviates from the training dataset.
* The models were noted to still face [hallucination](https://www.ibm.com/think/topics/ai-hallucinations).
* During training the models it was noted that the **Encoder-Decoder** (**Sequence-to-sequence**) models have a tendency to be unstable during training (The models kept crashing with a **NaN** error) specifically when using other pre-trained models.

## Conclusion
This project highlights the hypothetical pipeline of a system outlined in the <a href="/docs/(WIP) Technical Design Document v4.pdf" type="application/pdf">PDF</a>(Found here: **/docs/(WIP) Technical Design Document v4.pdf**) and its viability. The project implementation (code) proves such a system could work **IF AND ONLY IF (IFF)** every module performs its assigned task accurately, otherwise errors could cascade through the system impacting performance. This could potentially be resolved by explicity implementing quality control measures to identify and correct for any mistakes that could occur. In addition, hallucinations seem to be an issue that kept occuring in the generation process, this could be due to a flawed training objective or inadequate [Temperature](https://www.ibm.com/think/topics/llm-temperature) config value during the generation process. Regarding the instability issue when training Encoder-Decoder models, this could be remdied by pre-training a model and freezing the Encoder model during the finetuning phase, much like AI image generation models like [Stable Diffusion models](https://encord.com/blog/stable-diffusion-3-text-to-image-model/). More research could be performed on the suggested system's capabilities and performance with bigger and better datasets and models.
