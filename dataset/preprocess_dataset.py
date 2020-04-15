import json

with open('./data_conala/conala-test.json') as json_data:
    data_dict = json.load(json_data)
    print(data_dict)