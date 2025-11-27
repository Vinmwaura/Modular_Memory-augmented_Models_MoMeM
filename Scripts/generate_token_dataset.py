import os
import json
import pathlib
import argparse

from script_utils import load_json_data

def main():
    parser = argparse.ArgumentParser(
        description="Generates Token dataset from text dataset (Character-Based Tokenization).")

    parser.add_argument(
        "--dest-path",
        help="Destination output path for dataset json files.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--vocabulary-path",
        help="File path to Vocabulary.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--text-dataset-path",
        help="File path to text Dataset.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    vocabulary_path = args["vocabulary_path"]
    text_dataset_path = args["text_dataset_path"]

    os.makedirs(dest_path, exist_ok=True)

    # JSON Vocabulary.
    vocabulary_dict = load_json_data(json_fpath=vocabulary_path)

    # JSON text dataset.
    text_dataset_dict = load_json_data(json_fpath=text_dataset_path)

    # Token character(s) to token integer.
    tokens_to_id_dict = vocabulary_dict["tokens_to_id"]

    # List of all categories in dataset: "movie", "music", "occupation", etc.
    categories = text_dataset_dict["categories"]

    # Filepath to store categories.json
    categories_fpath = os.path.join(dest_path, "categories.json")
    try:
        with open(categories_fpath, "w") as json_f:
            json.dump({"data": categories}, json_f)

        print(f"Saved categories JSON files.")
    except Exception as e:
        raise e

    all_fpaths = []
    folder_index = 0
    len_text_dataset = len(text_dataset_dict["data"])
    for file_index, dataset_dict in enumerate(text_dataset_dict["data"]):
        print(f"{file_index + 1:,} / {len_text_dataset:,}")

        # Content tokens.
        content_characters = list(dataset_dict["content"])
        content_tokens = [tokens_to_id_dict[content_character] for content_character in content_characters]

        # Context tokens.
        context_tokens = {}
        for category in categories:
            context_tokens[category] = {}

            contexts = dataset_dict["context"][category]

            for context_type, context_data in contexts.items():
                context_characters = list(context_data)
                temp_context_tokens = [tokens_to_id_dict[context_character] for context_character in context_characters]

                context_tokens[category][context_type] = temp_context_tokens

        # Tags tokens.
        tags = dataset_dict["tags"]
        tags_characters = list(tags)
        tags_tokens = [tokens_to_id_dict[tags_character] for tags_character in tags_characters]

        temp_data_dict = {
            "tag": tags_tokens,
            "content": content_tokens,
            "context": context_tokens}

        curr_dir_path = os.path.join(dest_path, str(folder_index))
        os.makedirs(curr_dir_path, exist_ok=True)

        # JSON file.
        curr_file_path = os.path.join(curr_dir_path, str(file_index) + ".json")

        # List of all file paths.
        all_fpaths.append(curr_file_path)

        try:
            with open(curr_file_path, "w") as json_f:
                json.dump(temp_data_dict, json_f)

            print(f"Saved {curr_file_path} file.")
        except Exception as e:
            raise e

        if file_index % 1_000 == 0 and file_index > 0:
            folder_index += 1
    
    dataset_list = {
        "categories": categories,
        "fpaths": all_fpaths}

    file_list_fpath = os.path.join(dest_path, "token_dataset.json")
    try:
        with open(file_list_fpath, "w") as json_f:
            json.dump(dataset_list, json_f, indent=4)

        print(f"Saved JSON file.")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
