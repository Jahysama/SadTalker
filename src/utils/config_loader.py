import json


class Dict2Args(object):

    def __init__(self, json_path: str, json_merge: str = None):
        with open(json_path, 'r') as config_file:
            dict1 = json.load(config_file)
        if json_merge:
            with open(json_merge, 'r') as config_file:
                dict2 = json.load(config_file)
            dict1.update(dict2)
        for key in dict1:
            setattr(self, key, dict1[key])

