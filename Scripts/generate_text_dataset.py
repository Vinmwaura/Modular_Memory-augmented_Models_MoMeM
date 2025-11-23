import os
import csv
import json
import string
import random
import pathlib
import argparse

from script_utils import load_dict_data

# Generate JSON string dataset.
def generate_dataset(data_dict):
    def get_random_dataset():
        random_fname = random.choice(data_dict["fnames"])
        random_lname = random.choice(data_dict["lnames"])

        random_music = random.choice(data_dict["music"])
        random_movie = random.choice(data_dict["movies"])
        random_hobby = random.choice(data_dict["hobbies"])
        random_location = random.choice(data_dict["locations"])
        random_occupation = random.choice(data_dict["occupations"])
        random_university = random.choice(data_dict["universities"])

        # Used in template strings for dataset.
        format_dict = {
            "FName": random_fname,
            "LName": random_lname,
            "Occupation": random_occupation,
            "Location": random_location,
            "Movie": random_movie,
            "Music": random_music,
            "Hobbies": random_hobby,
            "University": random_university}
        
        return format_dict
    return get_random_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Generate text dataset using template structure.")

    parser.add_argument(
        "--dest-path",
        help="Destination output path for dataset json.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--lists-path",
        help="File path to CSV Lists.",
        required=False,
        default="./csv_List",
        type=pathlib.Path)
    parser.add_argument(
        "--template-path",
        help="File path to JSON Template.",
        required=False,
        default="./json_Template",
        type=pathlib.Path)
    parser.add_argument(
        "--num-training-data",
        help="Number of training dataset.",
        required=True,
        default=1,
        type=int)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    num_training_data = args["num_training_data"]
    list_path = args["lists_path"]
    template_path = args["template_path"]

    os.makedirs(dest_path, exist_ok=True)

    # CSV List.
    csvlist_dict = load_dict_data(list_path)

    # JSON Templates.
    template_fpath = os.path.join(template_path, "Dataset_template.json")
    with open(template_fpath) as json_f:
        template_json = json.load(json_f)

    # Generate Training dataset.
    dataset_generator = generate_dataset(data_dict=csvlist_dict)

    all_train_data = {
        "categories": template_json["context_categories"],
        "data": []}

    for train_data_index in range(num_training_data):
        print(f"Training dataset: {train_data_index + 1:,} / {num_training_data:,}")

        dataset = dataset_generator()

        content = template_json["content"].format(**dataset)

        content_fields = template_json["content_fields"]
        temp_content = [x.lstrip() for x in content.split(";")][:-1]
        content_dict = dict(zip(content_fields, temp_content))

        context_data = {}
        keys_list = list(template_json["context"].keys())
        for key in keys_list:
            random_item = random.choice(template_json["context"][key])
            context_data[key] = {
                "prompt": random_item["Prompt"].format(**dataset),
                "response": random.choice(random_item["Response"]).format(**dataset),
                "summary": content_dict[key]
            }

        all_train_data["data"].append(
            {
                "person_name": content_dict["person"],
                "content": content,
                "context": context_data
            })

    try:
        training_dataset_path = os.path.join(
            dest_path,
            "dataset.json")
        with open(training_dataset_path, "w") as f:
            json.dump(all_train_data, f, indent=4)

        print("Successfully saved training dataset.")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
