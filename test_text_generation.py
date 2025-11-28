import os
import csv
import json
import random
import pathlib
import argparse

import torch
import torch.nn.functional as F

from models.Transformer import Transformer

from utils.model_utils import load_model
from utils.generation_utils import generate_text

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.001:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "Text Generation - Transformer model(s)"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--temperature",
        help="Temperature parameter for softmax sampling.",
        type=restricted_float,
        default=1.0)
    parser.add_argument(
        "--vocabulary-path",
        help="File path to vocabulary.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-0-checkpoint",
        help="File path to model_0 checkpoint.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-1-checkpoint",
        help="File path to model_1 checkpoint.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-2-checkpoint",
        help="File path to model_2 checkpoint.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--test-data-path",
        help="File path to JSON data.",
        required=False,
        default="./test_data.json",
        type=pathlib.Path)
    parser.add_argument(
        "--template-path",
        help="File path to JSON Template.",
        required=False,
        default="./Scripts/json_Template/dataset_template.json",
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    temperature = args["temperature"]  # Temperature value.
    test_data_path = args["test_data_path"]  # JSON data.
    template_path = args["template_path"]  # JSON template.
    vocabulary_path = args["vocabulary_path"]  # Vocabulary json file path (*.json).
    model_0_checkpoint = args["model_0_checkpoint"]
    model_1_checkpoint = args["model_1_checkpoint"]
    model_2_checkpoint = args["model_2_checkpoint"]

    # List of model file paths.
    models_fpaths_list = [
        model_0_checkpoint,
        model_1_checkpoint,
        model_2_checkpoint]

    # Load Vocabulary / Dictionary dataset.
    with open(vocabulary_path, "r") as json_f:
        vocabulary_dict = json.load(json_f)
    
    # Token character to integer id.
    char_to_id_dict = vocabulary_dict["tokens_to_id"]

    # Token integer id to character id_to_tokens.
    id_to_char_dict = {}
    for token, token_id in char_to_id_dict.items():
        id_to_char_dict[token_id] = token

    # Special Tokens.
    special_tokens = vocabulary_dict["special_tokens_to_id"]
    special_tokens_list = list(special_tokens.values())

    # JSON Test Data dict.
    with open(test_data_path, "r") as test_json_f:
        format_dict = json.load(test_json_f)

    full_name = f"{format_dict["fname"]} {format_dict["lname"]}"

    # JSON Templates.
    with open(template_path) as json_f:
        template_json = json.load(json_f)

    context = template_json["context"]
    content = template_json["content"].format(**format_dict)
    content_fields = template_json["content_fields"]

    temp_content = [x.lstrip() for x in content.split(";")][:-1]
    content_dict = dict(zip(content_fields, temp_content))

    keys_list = list(context.keys())

    random_key = random.choice(keys_list)
    random_item = random.choice(context[random_key])

    model_str_dict = {
        "content": content,
        "prompt": random_item["prompt"].format(**format_dict),
        "context": content_dict[random_key]}

    print("*" * 100)
    print(f"Content => {model_str_dict["content"]}")
    print(f"Prompt => {model_str_dict["prompt"]}")
    print(f"Context (Ground Truth) => {model_str_dict["context"]}")
    print("*" * 100)

    # Convert Text string to Token ID Integers.
    model_token_dict = {}
    for key, val in model_str_dict.items():
        # Character-Based Tokenization.
        characters = list(val)
        model_token_dict[key] = [char_to_id_dict[character] for character in characters]
    
    # Prepend and append special tokens to the prompt token.
    input_prompt_tokens = \
        [special_tokens["start_prompt"]] +\
        model_token_dict["prompt"] +\
        [special_tokens["end_prompt"]]

    for model_type, model_checkpoint_fpath in enumerate(models_fpaths_list):
        print(f"Loading pre-trained Model_{model_type}.")
        print("*" * 30)

        classifier_status, classifier_dict = load_model(model_checkpoint_fpath)
        if not classifier_status:
            raise Exception(f"An error occured while loading pretrained Model_{model_type} checkpoint!")

        num_decoder_embeddings = classifier_dict["num_decoder_embeddings"]
        num_encoder_embeddings = classifier_dict["num_encoder_embeddings"]
        embedding_dim = classifier_dict["embedding_dim"]
        hidden_dim = classifier_dict["hidden_dim"]
        num_heads = classifier_dict["num_heads"]
        num_encoder_blocks = classifier_dict["num_encoder_blocks"]
        num_decoder_blocks = classifier_dict["num_decoder_blocks"]
        out_classes = classifier_dict["num_decoder_embeddings"]
        # use_cross_attn = classifier_dict["use_cross_attn"]
        activation_type = classifier_dict["activation_type"]

        use_cross_attn = False
        if model_type != 0:
            use_cross_attn = True

        # Transformer model.
        model = Transformer(
            special_tokens=special_tokens,
            num_decoder_embeddings=num_decoder_embeddings,
            num_encoder_embeddings=num_encoder_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            out_classes=num_decoder_embeddings,
            use_cross_attn=use_cross_attn,
            activation_type=activation_type)

        model.custom_load_state_dict(classifier_dict["model"])
        model = model.to(device)

        context_window = classifier_dict["context_window"]

        if model_type == 0:
            end_special_tokens = special_tokens["end_tag"]
            curr_prompt_tokens = input_prompt_tokens + [special_tokens["start_tag"]]
            encoder_prompt_tokens = None
        elif model_type == 1:
            end_special_tokens = special_tokens["end_summary"]
            curr_prompt_tokens = input_prompt_tokens + [special_tokens["start_summary"]]
            encoder_prompt_tokens = [special_tokens["start_encoding"]] + model_token_dict["content"] + [special_tokens["end_encoding"]]
        elif model_type == 2:
            end_special_tokens = special_tokens["end_response"]
            curr_prompt_tokens = input_prompt_tokens + [special_tokens["start_response"]]
            encoder_prompt_tokens = [special_tokens["start_encoding"]] + model_token_dict["context"] + [special_tokens["end_encoding"]]

        # Response from the model.
        model_response = generate_text(
            device=device,
            model=model,
            context_window=context_window,
            model_type=model_type,
            special_tokens=special_tokens_list,
            end_special_tokens=end_special_tokens,
            input_data=curr_prompt_tokens,
            encoder_data=encoder_prompt_tokens,
            inverted_vocabulary=id_to_char_dict,
            temperature=temperature)

        if model_type == 0:
            validity = "Found 1 result(s)" if model_response == full_name else "Found 0 result(s)"
            print(f"Model Output (Named-Entity Recognition) => \"{model_response}\"")
            print(f"{validity} (This is a mockup of a Named-Entity Recognition task.)")
        elif model_type == 1:
            validity = "Valid summary" if model_response == model_str_dict["context"] else "Invalid summary"
            print(f"Model Output (Summarization) => \"{model_response}\"")
            print(f"{validity} (This is a mockup of a Summary task.)")
        elif model_type == 2:
            validity = "Possibly correct" if format_dict[random_key] in model_response else "Incorrect"
            print(f"Model Output (Text Generation) => \"{model_response}\"")
            print(f"{validity} (This is a mockup of the Generation task.)")
        print("*" * 100)

if __name__ == "__main__":
    main()
