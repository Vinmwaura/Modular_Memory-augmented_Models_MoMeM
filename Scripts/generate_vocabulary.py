import os
import csv
import json
import string
import pathlib
import argparse

from script_utils import load_dict_data

def main():
    parser = argparse.ArgumentParser(
        description="Generate vocabulary using characters (ASCII printable) and words (Delineated by spaces).")

    parser.add_argument(
        "--dest-path",
        help="Destination output path for dataset json.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--lists-path",
        help="File path to CSV List.",
        required=False,
        default="./csv_List",
        type=pathlib.Path)
    parser.add_argument(
        "--template-path",
        help="File path to JSON Template.",
        required=False,
        default="./json_Template",
        type=pathlib.Path)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    list_path = args["lists_path"]
    template_path = args["template_path"]

    os.makedirs(dest_path, exist_ok=True)

    # CSV List Dict.
    sentences_list = []
    csvlist_dict = load_dict_data(list_path)
    
    # Get every unique character from the list of sentences.
    unique_characters = set()

    # Not elegant but works.
    for _, csv_list in csvlist_dict.items():
        for word in csv_list:
            for character in word:
                unique_characters.add(character)

    # Hack: Ensure all lowercase and uppercase ASCII characters are represented.
    for ascii_lowercase in string.ascii_lowercase:
        unique_characters.add(ascii_lowercase)
    for ascii_uppercase in string.ascii_uppercase:
        unique_characters.add(ascii_uppercase)

    # JSON Templates.
    temp_format_dict = {
        "FName": "",
        "LName": "",
        "Occupation": "",
        "Location": "",
        "Movie": "",
        "Music": "",
        "Hobbies": "",
        "University": ""}

    template_fpath = os.path.join(template_path, "Dataset_template.json")
    with open(template_fpath) as json_f:
        template_json = json.load(json_f)

    content_string = template_json["content"].format(**temp_format_dict)
    for content_word in content_string.split():
        unique_characters.add(content_word)

    for _, context_categories in template_json["context"].items():
        for context_dict in context_categories:
            prompt_text = context_dict["Prompt"].format(**temp_format_dict)
            for prompt_word in prompt_text.split():
                unique_characters.add(prompt_word)

            for response_text in context_dict["Response"]:
                response_text = response_text.format(**temp_format_dict)

                for response_word in response_text.split():
                    unique_characters.add(response_word)

    unique_characters_list = list(unique_characters)
    unique_characters_list.sort()

    # Token to integer id.
    vocabulary_data = {"tokens_to_id": {}}
    for index, unique_characters in enumerate(unique_characters_list):
        vocabulary_data["tokens_to_id"][unique_characters] = index

    """
    Special Tokens used in delineating the start and end of specific
    information as well as pad input.
    """
    len_vocabulary = len(vocabulary_data["tokens_to_id"])
    vocabulary_data["special_tokens_to_id"] = {
        "pad_token": len_vocabulary + 0,
        "start_prompt": len_vocabulary + 1,
        "end_prompt": len_vocabulary + 2,
        "start_tag": len_vocabulary + 3,
        "end_tag": len_vocabulary + 4,
        "SContext": len_vocabulary + 5,
        "EContext": len_vocabulary + 6,
        "start_response": len_vocabulary + 7,
        "end_response": len_vocabulary + 8}

    try:
        vocabulary_fpath = os.path.join(dest_path, "Vocabulary.json")
        with open(vocabulary_fpath, "w") as json_f:
            json.dump(vocabulary_data, json_f, indent=4)

        print("Successfully saved vocabulary!")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
