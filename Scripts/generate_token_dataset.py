import os
import json
import pathlib
import argparse

# Load JSON dataset.
def load_json_data(json_fpath):
    with open(json_fpath, "r") as json_f:
        data_dict = json.load(json_f)

    return data_dict

def main():
    parser = argparse.ArgumentParser(
        description="Generates token dataset from text (Tokenization).")

    parser.add_argument(
        "--dest-path",
        help="Destination output path for dataset json files.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--dictionary-path",
        help="File path to Dictionary.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--text-dataset-path",
        help="File path to text Dataset.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    dictionary_path = args["dictionary_path"]
    text_dataset_path = args["text_dataset_path"]

    os.makedirs(dest_path, exist_ok=True)

    # JSON Vocabulary / Dictionary.
    dictionary_dict = load_json_data(json_fpath=dictionary_path)

    # JSON text dataset.
    text_dataset_dict = load_json_data(json_fpath=text_dataset_path)

    # Token character(s) to token integer.
    dictionary_dict = dictionary_dict["tokens_to_id"]

    # List of all categories in dataset: "father", "mother", "likes" etc.
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
    for file_index, (person_name, person_data) in enumerate(text_dataset_dict["data"].items()):
        print(f"{file_index + 1:,} / {len_text_dataset:,}")

        # Content tokens.
        content_tokens = []
        split_contents = person_data["content"].split(" ")
        for index, word in enumerate(split_contents):
            if word in dictionary_dict:
                # Word-based tokenization (Used in template words).
                token = dictionary_dict[word]
                content_tokens.append(token)
            else:
                # Character-based tokenization (Used in names).
                characters = list(word)
                token = [dictionary_dict[character] for character in characters]
                content_tokens.extend(token)

            token = dictionary_dict[" "]

            if index < len(split_contents) - 1:
                content_tokens.append(token)

        # Context tokens.
        context_tokens = {}
        for category in categories:
            context_tokens[category] = {}

            contexts = person_data["context"][category]
            for context_type, context_data in contexts.items():
                split_contexts = context_data.split(" ")

                temp_context_tokens = []
                for index, word in enumerate(split_contexts):
                    if word in dictionary_dict:
                        token = dictionary_dict[word]
                        temp_context_tokens.append(token)
                    else:
                        characters = list(word)
                        token = [dictionary_dict[character] for character in characters]
                        temp_context_tokens.extend(token)

                    if index < len(split_contexts) - 1:
                        token = dictionary_dict[" "]
                        temp_context_tokens.append(token)

                context_tokens[category][context_type] = temp_context_tokens

        # Person First and Last Name tokens (Character-based tokenizations).
        name_characters = list(person_name)
        name_tokens = [dictionary_dict[name_character] for name_character in name_characters]

        curr_dir_path = os.path.join(dest_path, str(folder_index))
        os.makedirs(curr_dir_path, exist_ok=True)

        # JSON file.
        curr_file_path = os.path.join(curr_dir_path, person_name + ".json")

        # List of all file paths.
        all_fpaths.append(curr_file_path)

        temp_data_dict = {
            "tag": name_tokens,  # Full name == tags.
            "content": content_tokens,
            "context": context_tokens}

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

    file_list_fpath = os.path.join(dest_path, "DatasetList.json")
    try:
        with open(file_list_fpath, "w") as json_f:
            json.dump(dataset_list, json_f, indent=4)

        print(f"Saved JSON file.")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
