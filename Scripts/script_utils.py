import os
import csv
import json
from itertools import chain

# Load JSON dataset.
def load_json_data(json_fpath):
    with open(json_fpath, "r") as json_f:
        data_dict = json.load(json_f)

    return data_dict

# Loads list of items to be randomly selected for the template.
def load_csv_data(csv_fpath, delimiter='\n'):
    with open(csv_fpath) as csv_f:
        reader = csv.reader(csv_f, delimiter=delimiter)
        data_list = list(reader)

    # Flatten list of lists into a list of elements.
    data_list_flat = list(chain.from_iterable(data_list))
    return data_list_flat

# Load CSV list into memory.
def load_dict_data(list_path):
    data_list_dict = {}  # Dictionary of all lists.

    # First Name CSV Lists.
    fname_list_path = os.path.join(list_path, "First_Name.csv")
    data_list_dict["fname"] = load_csv_data(csv_fpath=fname_list_path)

    # Last Name CSV Lists.
    lname_list_path = os.path.join(list_path, "Last_Name.csv")
    data_list_dict["lname"] = load_csv_data(csv_fpath=lname_list_path)

    # Occupations CSV Lists.
    occupation_list_path = os.path.join(list_path, "Occupation.csv")
    data_list_dict["occupations"] = load_csv_data(csv_fpath=occupation_list_path)

    # Movies CSV Lists.
    movies_list_path = os.path.join(list_path, "Movies.csv")
    data_list_dict["movies"] = load_csv_data(csv_fpath=movies_list_path)

    # Music CSV Lists.
    music_list_path = os.path.join(list_path, "Music.csv")
    data_list_dict["music"] = load_csv_data(csv_fpath=music_list_path)

    # Hobbies CSV Lists.
    hobbies_list_path = os.path.join(list_path, "Hobbies.csv")
    data_list_dict["hobbies"] = load_csv_data(csv_fpath=hobbies_list_path)

    # Universities CSV Lists.
    universities_list_path = os.path.join(list_path, "Universities.csv")
    data_list_dict["universities"] = load_csv_data(csv_fpath=universities_list_path)

    # Locations (Cities) CSV Lists.
    locations_list_path = os.path.join(list_path, "Locations.csv")
    data_list_dict["locations"] = load_csv_data(csv_fpath=locations_list_path)

    return data_list_dict
