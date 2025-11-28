import os
import json
import random
import pathlib
import logging
import argparse

import torch
import torch.nn.functional as F

from models.Transformer import Transformer

from dataset_loader.token_dataset import TokenDataset

from utils.generation_utils import generate_text
from utils.model_utils import (
    save_model,
    load_model)

def checkpoint_model(
        data_dict,
        out_dir,
        model,
        model_optim,
        logging):
    global_steps = data_dict["global_steps"]

    # Save model that has achieved max TPR with the dataset.
    model_dict = {
        **data_dict,
        "model": model.state_dict(),
        "optimizer": model_optim.state_dict()}

    save_status = save_model(
        model_dict=model_dict,
        dest_path=out_dir,
        init_folder=True,
        file_name=f"{global_steps}_model.pt",
        logging=logging)
    if save_status is True:
        logging("Successfully saved model.")
    else:
        logging("Error occured saving model.")

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.001:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "(Decoder-only & Encoder-Decoder) Transformers"

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
        "--model-type",
        type=int,
        choices=[0, 1, 2],
        help="Choose the model to be trained (0: Model_0, 1: Model_1, 2: Model_2)")
    parser.add_argument(
        "--tr-dataset-path",
        help="File path to training json dataset file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--tst-dataset-path",
        help="File path to testing json dataset file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--batch-size",
        help="Batch size of dataset.",
        type=int,
        default=64)
    parser.add_argument(
        "--checkpoint-steps",
        help="Steps for checkpointing and/or testing model.",
        type=int,
        default=1_000)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint to load from (if any).",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "--lr-steps",
        help="Global steps in between halving learning rate.",
        default=50_000,
        type=int)
    parser.add_argument(
        "--load-optim",
        action='store_true',
        help="Load model's optimizer's weights and parameters, if loading model.")
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to JSON config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    temperature = args["temperature"]  # Temperature value.
    lr_steps = args["lr_steps"]  # Global steps in between halving learning rate.
    model_type = args["model_type"]  # 0: Model_0, 1: Model_1, 2: Model_2.
    load_optim = args["load_optim"]  # Reload saved optimizer weights.
    vocabulary_path = args["vocabulary_path"]  # Vocabulary json file path (*.json).
    tr_dataset_path = args["tr_dataset_path"]  # Training json file path (*.json).
    tst_dataset_path = args["tst_dataset_path"]  # Testing json file path (*.json).
    batch_size = args["batch_size"]  # Batch size of training dataset.
    model_checkpoint = args["model_checkpoint"]  # Filepath to models saved.
    checkpoint_steps = args["checkpoint_steps"]  # Steps to checkpoint model.
    out_dir = args["out_dir"]  # Destination path for model.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    # Load Vocabulary dataset.
    # TODO: Use separate, appropriate vocabularies for each model type.
    with open(vocabulary_path, "r") as json_f:
        vocabulary_dict = json.load(json_f)

    # Inverted vocabulary: id_to_tokens.
    inverted_vocabulary = {}
    for token, token_id in vocabulary_dict["tokens_to_id"].items():
        inverted_vocabulary[token_id] = token

    # Special Tokens.
    special_tokens = vocabulary_dict["special_tokens_to_id"]
    special_tokens_list = list(special_tokens.values())

    # Config Dict.
    config_json = args["config_path"]  # Load and Parse config JSON.
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    # Training params.
    global_steps = 0
    max_global_steps = config_dict["max_global_steps"]

    # Model Params (From config file).
    lr_gamma = config_dict["lr_gamma"]  # Learning Rate gamma value.
    model_lr = config_dict["model_lr"]
    num_heads = config_dict["num_heads"]
    hidden_dim = config_dict["hidden_dim"]
    embedding_dim = config_dict["embedding_dim"]
    context_window = config_dict["context_window"]
    activation_type = config_dict["activation_type"]
    num_encoder_blocks = None
    if model_type != 0:
        num_encoder_blocks = config_dict["num_encoder_blocks"]
    num_decoder_blocks = config_dict["num_decoder_blocks"]

    # Datasets.
    tr_dataset = TokenDataset(
        json_dataset=tr_dataset_path,
        context_window=context_window,
        special_tokens=special_tokens)
    tst_dataset = TokenDataset(
        json_dataset=tst_dataset_path,
        context_window=context_window,
        special_tokens=special_tokens)

    # Test Dataset filepaths.
    tst_categories = tst_dataset.categories
    tst_fpaths_list = tst_dataset.fpaths_list

    # Dataloaders.
    tr_dataloader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True)
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True)
    tst_iterator = iter(tst_dataloader)

    """
    Transformer Model Architecture.
    > Model_0: Decoder-only models (Input_Encoder: None | Input_Decoder: Prompt, Tags).
    > Model_1: Encoder-Decoder models (Input_Encoder: Content | Input_Decoder: Prompt, Summary).
    > Model_2: Encoder-Decoder models (Input_Encoder: Summary | Input_Decoder: Prompt, Response).
    """
    if model_type == 0:
        # Decoder-only Model (Model_0).
        use_cross_attn = False
        num_encoder_embeddings = None
    else:
        # Encoder-Decoder Model (Model_1 & Model_2).
        use_cross_attn = True
        num_encoder_embeddings = len(inverted_vocabulary) + len(special_tokens) + 1  # Includes [Pad] token.

    num_decoder_embeddings = len(inverted_vocabulary) + len(special_tokens)

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

    # Load Transformer Model checkpoints if any.
    if model_checkpoint is not None:
        logging.info(f"Loading pretrained Model_{model_type}...")
        classifier_status, classifier_dict = load_model(model_checkpoint)
        if not classifier_status:
            raise Exception(f"An error occured while loading pretrained Model_{model_type} checkpoint!")

        model.custom_load_state_dict(classifier_dict["model"])
        model = model.to(device)

        model_optim = torch.optim.Adam(
            model.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

        # Load Optimizer params and global steps params.
        if load_optim:
            global_steps = classifier_dict["global_steps"]

            logging.info("Resuming training using saved optimizer weights and global_steps...")
            model_optim.load_state_dict(classifier_dict["optimizer"])
    else:
        model = model.to(device)

        model_optim = torch.optim.Adam(
            model.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

    # Model parameter size.
    model_params_size = sum(param.numel() for param in model.parameters())

    # Learning Rate Scheduler.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=model_optim,
        step_size=lr_steps,
        gamma=lr_gamma)

    # https://pytorch.org/docs/stable/amp.html
    scaler = torch.cuda.amp.GradScaler()

    # Log file path.
    log_path = os.path.join(
        out_dir,
        f"{project_name}.log")

    # Logs Info to parent directory.
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG,
        force=True)

    logging.info(f"{project_name}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("#" * 100)
    logging.info("Dataset Parameters.")
    if model_type != 0:
        logging.info(f"Encoder Vocab Size: {num_encoder_embeddings:,}")
    logging.info(f"Decoder Vocab Size: {num_decoder_embeddings:,}")
    logging.info(f"Total Train Dataset: {len(tr_dataset):,}")
    logging.info(f"Context window: {context_window:,}")
    logging.info(f"Train Batch Size: {batch_size:,}")
    logging.info("#" * 100)
    logging.info("Model Parameters.")
    logging.info(f"Temperature: {temperature:.3f}")
    logging.info(f"Total Model Param size: {model_params_size:,}")
    logging.info(f"Number of heads: {num_heads:,}")
    if model_type != 0:
        logging.info(f"Number of Encoder Blocks: {num_encoder_blocks:,}")
    logging.info(f"Number of Decoder Blocks: {num_decoder_blocks:,}")
    logging.info(f"Embedding Dimension: {embedding_dim:,}")
    logging.info(f"Hidden Dimension: {hidden_dim:,}")
    logging.info(f"Activation Type: {activation_type}")
    logging.info(f"Model Learning Rate: {model_optim.param_groups[0]['lr']:,}")
    logging.info(f"Model Learning Rate gamma: {lr_gamma:,}")
    logging.info("#" * 100)
    logging.info("Training Parameters.")
    logging.info(f"Step: {global_steps:,}")
    logging.info(f"Max Global step: {max_global_steps:,}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps:,}")
    logging.info("#" * 100)

    model_data_dict = {
        "lr_gamma": lr_gamma,
        "num_heads": num_heads,
        "model_type": model_type,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "global_steps": global_steps,
        "embedding_dim": embedding_dim,
        "use_cross_attn": use_cross_attn,
        "context_window": context_window,
        "activation_type": activation_type,
        "num_encoder_blocks": num_encoder_blocks,
        "num_decoder_blocks": num_decoder_blocks,
        "num_encoder_embeddings": num_encoder_embeddings,
        "num_decoder_embeddings": num_decoder_embeddings}

    # Training starts here.
    stop_training = False
    while not stop_training:
        for index, tr_data_dict in enumerate(tr_dataloader):
            # Checkpoint and test model.
            if global_steps % checkpoint_steps == 0:
                model_data_dict["global_steps"] = global_steps
                checkpoint_model(
                    data_dict=model_data_dict,
                    out_dir=out_dir,
                    model=model,
                    model_optim=model_optim,
                    logging=logging.info)

                # Generate text using randomly selected testing dataset and model.
                random_tst_fpath = random.choice(tst_fpaths_list)
                with open(random_tst_fpath, "r") as json_f:
                    tst_json_data = json.load(json_f)

                # Context Tokens.
                context_dict = tst_json_data["context"]

                # Randomly pick a category for the prompt.
                random_category = random.choice(tst_categories)

                # Prompt.
                prompt_tokens = context_dict[random_category]["prompt"]
                prompt_token_list = [inverted_vocabulary[token_id] for token_id in prompt_tokens]
                prompt_text = "".join(prompt_token_list)

                logging.info("*" * 100)
                logging.info(f"Prompt text: {prompt_text}\n")

                # Prepend and append special tokens to the prompt token.
                input_prompt_tokens = [special_tokens["start_prompt"]] + prompt_tokens + [special_tokens["end_prompt"]]
                if model_type == 0:
                    end_special_tokens = special_tokens["end_tag"]
                    input_prompt_tokens = input_prompt_tokens + [special_tokens["start_tag"]]
                    encoder_prompt_tokens = None
                elif model_type == 1:
                    end_special_tokens = special_tokens["end_summary"]
                    input_prompt_tokens = input_prompt_tokens + [special_tokens["start_summary"]]
                    encoder_prompt_tokens = [special_tokens["start_encoding"]] + tst_json_data["content"] + [special_tokens["end_encoding"]]
                elif model_type == 2:
                    end_special_tokens = special_tokens["end_response"]
                    input_prompt_tokens = input_prompt_tokens + [special_tokens["start_response"]]
                    encoder_prompt_tokens = [special_tokens["start_encoding"]] + tst_json_data["context"][random_category]["summary"] + [special_tokens["end_encoding"]]

                # Conditional text passed as conditional input to the Encoder model.
                if encoder_prompt_tokens != None:
                    encoder_prompt_token_list = [inverted_vocabulary[token_id] for token_id in encoder_prompt_tokens[1:-1]]
                    encoder_prompt_txt = "".join(encoder_prompt_token_list)

                    logging.info(f"Encoder data (Conditional): {encoder_prompt_txt}\n")

                # Response from the model.
                model_response = generate_text(
                    device=device,
                    model=model,
                    context_window=context_window,
                    model_type=model_type,
                    special_tokens=list(special_tokens.values()),
                    end_special_tokens=end_special_tokens,
                    input_data=input_prompt_tokens,
                    encoder_data=encoder_prompt_tokens,
                    inverted_vocabulary=inverted_vocabulary,
                    temperature=temperature)

                logging.info(f"Model Response: {model_response}\n")
                logging.info("*" * 100)

            # Training Dataset (In, Target).
            tr_in_seq = tr_data_dict[f"model_{model_type}"]["in"]
            tr_in_seq = tr_in_seq.to(device)  # (N, Seq_dec)

            tr_target_seq = tr_data_dict[f"model_{model_type}"]["target"]
            tr_target_seq = tr_target_seq.to(device)  # (N, Seq_dec)

            if model_type == 0:
                tr_encoder_seq = None
            else:
                tr_encoder_seq = tr_data_dict[f"model_{model_type}"]["encoder"]
                tr_encoder_seq = tr_encoder_seq.to(device)  # (N,Seq_enc)

            model.train(mode=True)

            # Train Classifier.
            # Runs the forward pass under ``autocast``.
            with torch.autocast(device_type=device, dtype=torch.float16):
                tr_out_classifier = model(tr_in_seq, tr_encoder_seq)

                tr_target_seq_flat = tr_target_seq.flatten()  # (N*Seq,)
                tr_out_classifier_flat = tr_out_classifier.flatten(
                    start_dim=0,
                    end_dim=1)  # (N*Seq,Class)

                tr_classifier_loss = F.cross_entropy(
                    tr_out_classifier_flat,
                    tr_target_seq_flat,
                    ignore_index=special_tokens["pad_token"])
                if torch.isnan(tr_classifier_loss):
                    raise Exception("NaN encountered during training.")

            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(tr_classifier_loss).backward()

            scaler.step(model_optim)

            # Updates the scale for next iteration.
            scaler.update()

            model_optim.zero_grad()

            train_classifier_loss = tr_classifier_loss.item()

            # Test Classifier.
            try:
                tst_data_dict = next(tst_iterator)
            except StopIteration:
                tst_iterator = iter(tst_dataloader)
                tst_data_dict = next(tst_iterator)

            tst_in_seq = tst_data_dict[f"model_{model_type}"]["in"]
            tst_in_seq = tst_in_seq.to(device)  # (N, Seq_dec)

            tst_target_seq = tst_data_dict[f"model_{model_type}"]["target"]
            tst_target_seq = tst_target_seq.to(device)  # (N, Seq_dec)

            if model_type == 0:
                tst_encoder_seq = None
            else:
                tst_encoder_seq = tst_data_dict[f"model_{model_type}"]["encoder"]
                tst_encoder_seq = tst_encoder_seq.to(device)  # (N,Seq_enc)

            model.eval()

            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
                tst_out_classifier = model(tst_in_seq, tst_encoder_seq)

                tst_target_seq_flat = tst_target_seq.flatten()  # (N*Seq,)
                tst_out_classifier_flat = tst_out_classifier.flatten(
                    start_dim=0,
                    end_dim=1)  # (N*Seq,Class)

                tst_classifier_loss = F.cross_entropy(
                    tst_out_classifier_flat,
                    tst_target_seq_flat,
                    ignore_index=special_tokens["pad_token"])

            test_classifier_loss = tst_classifier_loss.item()

            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | Train Classifier Loss: {:,.5f} | Test Classifier Loss: {:,.5f} | LR: {:.3E}".format(
                global_steps + 1,
                index + 1,
                len(tr_dataloader),
                train_classifier_loss,
                test_classifier_loss,
                model_optim.param_groups[0]['lr'])

            logging.info(message)

            global_steps = global_steps + 1

            # Stop training when stopping criteria is met.
            if global_steps >= max_global_steps:
                stop_training = True
                break

            lr_scheduler.step()

        model_data_dict["global_steps"] = global_steps
        checkpoint_model(
            data_dict=model_data_dict,
            out_dir=out_dir,
            model=model,
            model_optim=model_optim,
            logging=logging.info)

if __name__ == "__main__":
    main()
