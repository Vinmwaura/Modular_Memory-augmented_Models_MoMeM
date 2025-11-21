import os
import csv
import json
import string
import random
import pathlib
import argparse

from itertools import chain

from faker import Faker
from faker.providers import DynamicProvider

Faker.seed(69)

# Loads list of items to be randomly selected for the template.
def load_data(csv_fpath, delimiter='\n'):
    with open(csv_fpath) as csv_f:
        reader = csv.reader(csv_f, delimiter=delimiter)
        data_list = list(reader)

    data_list_flat = list(chain.from_iterable(data_list))
    return data_list_flat

# Creates a custom provider without defining a new class.
def get_dynamic_provider(provider_name, elements_list):
    dynamic_provider = DynamicProvider(
        provider_name=provider_name,
        elements=elements_list)
    return dynamic_provider

# Load all csv list.
def load_all_list(list_path):
    likes_list_path = os.path.join(list_path, "Likes.csv")
    dislikes_list_path = os.path.join(list_path, "Dislikes.csv")
    hobbies_list_path = os.path.join(list_path, "Hobbies.csv")
    universities_list_path = os.path.join(list_path, "Universities.csv")
    locations_list_path = os.path.join(list_path, "Locations.csv")

    person_likes_list = load_data(csv_fpath=likes_list_path)
    person_dislikes_list = load_data(csv_fpath=dislikes_list_path)
    person_hobbies_list = load_data(csv_fpath=hobbies_list_path)
    universities_list = load_data(csv_fpath=universities_list_path)
    locations_list = load_data(csv_fpath=locations_list_path)

    data_list_dict = {
        "person_likes": person_likes_list,
        "person_dislikes": person_dislikes_list,
        "person_hobbies": person_hobbies_list,
        "universities": universities_list,
        "locations": locations_list}
    return data_list_dict

def generate_dataset(data_generator, data_json):
    person_FName = data_generator.first_name()

    # Hack: Ensures no duplicates, Faker runs out of names easily. 
    uppercase_letters = string.ascii_uppercase
    random_uppercase_character = random.choice(uppercase_letters)

    lowercase_letters = string.ascii_lowercase
    random_lowercase_string = ''.join(random.choice(lowercase_letters) for _ in range(5))

    # family_LName = data_generator.last_name()
    family_LName = random_uppercase_character + random_lowercase_string

    job = data_generator.job()
    father_FName = data_generator.first_name_male()
    mother_FName = data_generator.first_name_female()

    # Randomly picks item from CSV list.
    person_likes = data_generator.person_likes()
    person_dislikes = data_generator.person_dislikes()
    person_locations = data_generator.locations()
    person_hobbies = data_generator.person_hobbies()
    university_attended = data_generator.universities()

    # Used in template strings for dataset.
    format_dict = {
        "FName": person_FName,
        "LName": family_LName,
        "Occupation": job,
        "Location": person_locations,
        "Likes": person_likes,
        "Dislikes": person_dislikes,
        "Hobbies": person_hobbies,
        "University": university_attended,
        "Mother_FName": mother_FName,
        "Mother_LName": family_LName,
        "Father_FName": father_FName,
        "Father_LName": family_LName}

    content = data_json["Content"].format(**format_dict)

    content_fields = data_json["Content_fields"]
    temp_content = [x.lstrip() for x in content.split(";")][:-1]
    content_dict = dict(zip(content_fields, temp_content))

    context_data = {}
    keys_list = list(data_json["Context"].keys())
    for key in keys_list:
        random_item = random.choice(data_json["Context"][key])
        context_data[key] = {
            "prompt": random_item["Prompt"].format(**format_dict),
            "response": random.choice(random_item["Response"]).format(**format_dict),
            "summary": content_dict[key]
        }

    data_dict = {
        "content": content,
        "context": context_data}

    return content_dict["person"], data_json["Context_categories"], data_dict

# Faker generator.
def get_generator(data_list_dict):
    data_generator = Faker()
    for provider_name, elements in data_list_dict.items():
        temp_provider = DynamicProvider(
            provider_name=provider_name,
            elements=elements)
        data_generator.add_provider(temp_provider)

    return data_generator

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
    parser.add_argument(
        "--num-testing-data",
        help="Number of testing dataset.",
        required=True,
        default=1,
        type=int)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    num_training_data = args["num_training_data"]
    num_testing_data = args["num_testing_data"]
    list_path = args["lists_path"]
    template_path = args["template_path"]

    os.makedirs(dest_path, exist_ok=True)

    # CSV List.
    data_list_dict = load_all_list(list_path)

    # JSON Templates.
    template_fpath = os.path.join(template_path, "Dataset_template.json")
    with open(template_fpath) as json_f:
        data_json = json.load(json_f)

    # Faker Generator.
    data_generator = get_generator(data_list_dict=data_list_dict)

    # Generate Training dataset.
    all_train_data = None
    for train_data_index in range(num_training_data):
        print(f"Training dataset: {train_data_index + 1:,} / {num_training_data:,}")

        person_name, categories, data_dict = generate_dataset(
            data_generator=data_generator,
            data_json=data_json)

        if all_train_data is None:
            all_train_data = {
                "categories": categories,
                "data": {
                    person_name: data_dict
                }
            }
        else:
            all_train_data["data"][person_name] = data_dict

    try:
        training_dataset_path = os.path.join(
            dest_path,
            "Train.json")
        with open(training_dataset_path, "w") as f:
            json.dump(all_train_data, f, indent=4)

        print("Successfully saved training dataset.")
    except Exception as e:
        raise e

    # Generate Testing dataset.
    all_test_data = None

    for test_data_index in range(num_testing_data):
        print(f"Testing dataset: {test_data_index + 1:,} / {num_testing_data:,}")

        person_name, categories, data_dict = generate_dataset(
            data_generator=data_generator,
            data_json=data_json)

        if all_test_data is None:
            all_test_data = {
                "categories": categories,
                "data": {
                    person_name: data_dict
                }
            }
        else:
            all_test_data["data"][person_name] = data_dict

    try:
        testing_dataset_path = os.path.join(
            dest_path,
            "Test.json")
        with open(testing_dataset_path, "w") as f:
            json.dump(all_test_data, f, indent=4)

        print("Successfully saved testing dataset.")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
