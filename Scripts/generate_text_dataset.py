import os
import csv
import json
import string
import random
import pathlib
import argparse

from script_utils import load_dict_data

# Generate JSON string dataset.
def generate_dataset(data_dict, prob_threshold=0.5):
    def get_random_dataset():
        random_fname = random.choice(data_dict["fname"])
        random_lname = random.choice(data_dict["lname"])
        random_music = random.choice(data_dict["music"])
        random_movie = random.choice(data_dict["movies"])
        random_hobby = random.choice(data_dict["hobbies"])
        random_location = random.choice(data_dict["locations"])
        random_occupation = random.choice(data_dict["occupations"])
        random_university = random.choice(data_dict["universities"])

        """
        HACK: Randomly augment dataset by shuffling the characters
        prob_threshold: Original characters
        1 - prob_threshold: Shuffled characters.
        """
        # Used in template strings for dataset.
        format_dict = {
            "fname": random_fname if random.random() <= prob_threshold else "".join(random.sample(random_fname, len(random_fname))),
            "lname": random_lname if random.random() <= prob_threshold else "".join(random.sample(random_lname, len(random_lname))),
            "occupation": random_occupation,
            "location": random_location,
            "movie": random_movie,
            "music": random_music,
            "hobbies": random_hobby,
            "university": random_university}

        return format_dict
    return get_random_dataset

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range 0.0 < x < 1.0"%(x,))
    return x

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
        "--context-window",
        help="Context window (Character-Based Tokens).",
        default=320,
        type=int)
    parser.add_argument(
        "--num-datapoints",
        help="Size of dataset in terms of records.",
        required=True,
        default=1,
        type=int)
    parser.add_argument(
        "--prob-threshold",
        help="Probability threshold used to augment dataset with shuffled characters.",
        type=restricted_float,
        default=0.5)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    list_path = args["lists_path"]
    template_path = args["template_path"]
    num_datapoints = args["num_datapoints"]
    prob_threshold = args["prob_threshold"]
    context_window = args["context_window"]

    os.makedirs(dest_path, exist_ok=True)

    # CSV List.
    csvlist_dict = load_dict_data(list_path)

    # JSON Templates.
    template_fpath = os.path.join(template_path, "dataset_template.json")
    with open(template_fpath) as json_f:
        template_json = json.load(json_f)

    # Generate Training dataset.
    dataset_generator = generate_dataset(
        data_dict=csvlist_dict,
        prob_threshold=prob_threshold)

    all_train_data = {
        "categories": template_json["context_categories"],
        "data": []}

    # Prevents infinite loops.
    curr_loop = 0
    max_loop = int(1e9)

    datapoints_index = 0
    while datapoints_index < num_datapoints:
        dataset = dataset_generator()

        tags = f"{dataset['fname']} {dataset['lname']}"
        if len(tags) > context_window:
            continue

        content = template_json["content"].format(**dataset)
        if (len(content) + 2) > context_window:  # Includes [Start] and [End] Special Tokens.
            continue

        content_fields = template_json["content_fields"]
        temp_content = [x.lstrip() for x in content.split(";")][:-1]
        content_dict = dict(zip(content_fields, temp_content))

        continue_context_loop = False

        context_data = {}
        keys_list = list(template_json["context"].keys())
        for key in keys_list:
            random_item = random.choice(template_json["context"][key])

            prompt_text = random_item["prompt"].format(**dataset)
            response_text = random.choice(random_item["response"]).format(**dataset)
            summary_text = content_dict[key]

            len_prompt = len(prompt_text)
            len_summary = len(summary_text)
            len_response = len(response_text)

            # Module_2 Token input / output size (Includes all special tokens).
            if (len_prompt + len_summary + 3) > context_window:
                continue_context_loop = True
                break

            # Module_3 Token input / output size (Includes all special tokens).
            if (len_prompt + len_response + 3) > context_window:
                continue_context_loop = True
                break

            context_data[key] = {
                "prompt": prompt_text,
                "response": response_text,
                "summary": summary_text}

        # Skip loop, large token size input.
        if continue_context_loop:
            continue

        all_train_data["data"].append({
            "tags": tags,
            "content": content,
            "context": context_data
        })

        print(f"Datapoints: {datapoints_index + 1:,} / {num_datapoints:,}")
        datapoints_index += 1

        # HACK: Breaks out of infinite loops after a certain loop count.
        curr_loop += 1
        if curr_loop > max_loop:
            print(f"Exceeded maximum loop: {max_loop:,}")
            break

    # Save data in JSON file.
    try:
        training_dataset_path = os.path.join(
            dest_path,
            "text_dataset.json")
        with open(training_dataset_path, "w") as f:
            json.dump(all_train_data, f, indent=4)

        print("Successfully saved training dataset.")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
