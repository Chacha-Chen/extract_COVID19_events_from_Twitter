
import json
import pickle

def saveToPickleFile(save_object, save_file):
    with open(save_file, "wb") as pickle_out:
        pickle.dump(save_object, pickle_out)

def loadFromPickleFile(pickle_file):
    with open(pickle_file, "rb") as pickle_in:
        return pickle.load(pickle_in)

def saveToJSONFile(save_dict, save_file):
    with open(save_file, 'w') as fp:
        json.dump(save_dict, fp)

def loadFromJSONFile(json_file):
    with open(json_file, 'r') as fp:
        return json.load(fp)
