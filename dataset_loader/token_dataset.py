import json
import random

import torch
from torch.utils.data import Dataset

"""
Load Token Dataset for Model_0, Model_1, Model_2.
"""
class TokenDataset(Dataset):
    def __init__(self, json_dataset, context_window, special_tokens):
        self.special_tokens = special_tokens
        self.context_window = context_window

        with open(json_dataset, "r") as json_f:
            dataset = json.load(json_f)

        self.categories = dataset["categories"]
        self.fpaths_list = dataset["fpaths"]

    def __len__(self):
        return len(self.fpaths_list)

    def __getitem__(self, index):
        json_fpath = self.fpaths_list[index]

        with open(json_fpath, "r") as json_f:
            json_data = json.load(json_f)

        # Tag Tokens.
        tag_tokens = json_data["tag"]

        # Content Tokens.
        content_tokens = json_data["content"]

        # Pad Content to be of uniform length.
        content_paddings = [self.special_tokens["pad_token"]] * (self.context_window - (len(content_tokens) + 2))
        content_tokens_padded = [self.special_tokens["start_encoding"]] + content_tokens[:] + [self.special_tokens["end_encoding"]] + content_paddings

        # Context Tokens.
        context_dict = json_data["context"]

        # Randomly pick a category: contains a prompt, response, and summary.
        random_category = random.choice(self.categories)

        prompt_tokens = context_dict[random_category]["prompt"]
        summary_tokens = context_dict[random_category]["summary"]
        response_tokens = context_dict[random_category]["response"]

        # Pad Summaries to be of uniform length.
        summary_tokens_paddings = [self.special_tokens["pad_token"]] * (self.context_window - (len(summary_tokens) + 2))
        summary_tokens_padded = [self.special_tokens["start_encoding"]] + summary_tokens[:] + [self.special_tokens["end_encoding"]] + summary_tokens_paddings

        # Input and Target tokens for Model_0.
        combined_prompt_tokens = [self.special_tokens["start_prompt"]] + prompt_tokens + [self.special_tokens["end_prompt"]]
        combined_tag_tokens = [self.special_tokens["start_tag"]] + tag_tokens + [self.special_tokens["end_tag"]]
        model_0_tokens = combined_prompt_tokens + combined_tag_tokens

        # Pad Model_0 tokens to be of uniform length.
        model_0_paddings = [self.special_tokens["pad_token"]] * (self.context_window - len(model_0_tokens) + 1)
        in_model_0_tokens = model_0_tokens[:-1] + model_0_paddings
        target_model_0_tokens = model_0_tokens[1:] + model_0_paddings

        # Input and Target tokens for Model_1.
        combined_summary_tokens = [self.special_tokens["start_summary"]] + summary_tokens + [self.special_tokens["end_summary"]]
        model_1_tokens = combined_prompt_tokens + combined_summary_tokens

        # Pad token list to be of uniform length.
        model_1_paddings = [self.special_tokens["pad_token"]] * (self.context_window - len(model_1_tokens) + 1)
        in_model_1_tokens = model_1_tokens[:-1] + model_1_paddings
        target_model_1_tokens = model_1_tokens[1:] + model_1_paddings

        # Input and Target tokens for Model_2.
        combined_response_tokens = [self.special_tokens["start_response"]] + response_tokens + [self.special_tokens["end_response"]]
        model_2_tokens = combined_prompt_tokens + combined_response_tokens

        # Pad token list to be of uniform length.
        model_2_paddings = [self.special_tokens["pad_token"]] * (self.context_window - len(model_2_tokens) + 1)
        in_model_2_tokens = model_2_tokens[:-1] + model_2_paddings
        target_model_2_tokens = model_2_tokens[1:] + model_2_paddings

        tensor_dict = {
            "model_0": {
                "in": torch.tensor(in_model_0_tokens).long(),
                "target": torch.tensor(target_model_0_tokens).long()
            },
            "model_1": {
                "in": torch.tensor(in_model_1_tokens).long(),
                "target": torch.tensor(target_model_1_tokens).long(),
                "encoder": torch.tensor(content_tokens_padded).long() 
            },
            "model_2": {
                "in": torch.tensor(in_model_2_tokens).long(),
                "target": torch.tensor(target_model_2_tokens).long(),
                "encoder": torch.tensor(summary_tokens_padded).long()
            },
        }

        return tensor_dict
