import sys
import os
import io
import csv
import json
import ast
import re
import time
import math
from operator import add
from random import shuffle
from random import sample

# filter patterns with given pattern
# this is only used for API replace
def filter(feature_file_name, pattern, body_API):
    filtered = []
    print(feature_file_name)
    csv.field_size_limit(sys.maxsize)
    with open(feature_file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        header = next(csv_reader, None)
        line_count = 0
        # tokens\ttags\tRepAPI\tRetAPI\tBodyAPI\tfile\tline
        for row in csv_reader:
            # seq = row[0] # json string for gnn tokens
            # tags = set(row[1].split(',')) # separated with ',' when there are more than one
            repAPI = row[2] # with the form of API1::API2
            retAPI = set(row[3].split(',')) # separated with ',' when there are more than one
            bodyAPI = row[4] # separated with ',' when there are more than one
            # fname = row[5] # absolute file name of the pattern
            # line = int(row[6]) # line number
            if not body_API:
                if repAPI == pattern:
                    filtered.append(row)
            else:
                if pattern in retAPI or pattern in bodyAPI:
                    filtered.append(row)
    return filtered

# load pre-trained word embedding
def load_embedding():
    dic = {}
    with open('embeddings.txt', 'r') as f:
        for line in f.readlines():
            tokens = line.split('\t')
            dic[tokens[0].rstrip()] = [float(value) for value in tokens[1].split()]
    return dic

# embed a list of tokens, and sum embeddings
def embed_token_list(dic, token_list):
    feature_sum = [0] * len(dic['UNK'])
    # print(token_list)
    # print('before', feature_sum)
    for token in token_list:
        if token in dic :
            feature_sum = list( map(add, feature_sum, dic[token]) )
        else :
            print(token, 'not')
    if sum(feature_sum) == 0 :
        feature_sum = dic['UNK']
    # print('after', feature_sum)
    return feature_sum

# split token into words according to camel case
def word_split(token):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token.decode("utf-8"))
    return [m.group(0).lower() for m in matches]

# embed a string, which may be a camel case
def embed_string(dic, string, default_value):
    features = embed_token_list(dic, word_split(string.encode("utf-8")))
    if features == None :
        features = embed_token_list(dic, word_split(default_value))
    if features == None :
        print('should not!')
        features = dic['UNK']
    return features

# embed value type info
def embed_type(dic, type_str, default_value):
    # need to preprocess type info, e.g., java.util.List<java.lang.String> -> List
    index = type_str.find('<')
    if index > 0:
        type_str = type_str[:index]
    type_str = type_str[type_str.rfind('.') + 1:]
    return embed_string(dic, type_str, default_value)
    

default_value_map = {'kind': 'UnknownNode', 'op': 'UnknownUse', 'type': 'UnknownType', 'api': 'UnknownAPI'}

# embed features
def embedding(features):
    dic = load_embedding()
    converted = []
    for row in features:
        seq = row[0]
        obj = json.loads(seq) # {
                              #     "targets": [int,int],
                              #     "graph":[[int, int, int], []],
                              #     "node_features":[[Str, Str, Str, Str], []]
                              # }
        convertedFeature = {}
        for key, value in obj.items():
            if key == 'node_features':
                items = []
                for item in value:
                    comb_feature = []
                    comb_feature.extend(embed_string(dic, item[0], default_value_map['kind']))
                    comb_feature.extend(embed_string(dic, item[1], default_value_map['op']))
                    comb_feature.extend(embed_type(dic, item[2], default_value_map['type']))
                    comb_feature.extend(embed_string(dic, item[3], default_value_map['api']))
                    items.append(comb_feature)
                convertedFeature[key] = items
            else :
                convertedFeature[key] = value
        # convertedFeature['API'] = row[2]
        converted.append(convertedFeature)
    return converted

def split_list(a_list):
    shuffle(a_list)
    cut = len(a_list)//5
    return a_list[:cut], a_list[cut:]

def chunks(a_list, n):
    """Yield successive n-sized chunks from l."""
    sub_len = int(math.ceil(len(a_list) / n))
    for i in range(0, len(a_list), sub_len):
        yield a_list[i:i + sub_len]


def process(top_path, api, pattern, body_api):
    # files = ['correct.txt', 'wrong.txt']
    correct_files = ['11-17correct.txt', '1811-12correct.txt', '18correct.txt']
    wrong_files = ['11-17wrong.txt', '1811-12wrong.txt', '18wrong.txt']

    converted_string = []
    path = top_path + '/FixRuleMiner'

    for f in correct_files:
        filtered = filter(os.path.join(path, f), pattern, body_api)
        converted = embedding(filtered)
        converted_string.extend([str(ast.literal_eval(json.dumps(item))).replace("'", "\"") for item in converted])
    
    count_correct = len(converted_string)

    for f in wrong_files:
        filtered = filter(os.path.join(path, f), pattern, body_api)
        converted = embedding(filtered)
        converted_string.extend([str(ast.literal_eval(json.dumps(item))).replace("'", "\"") for item in converted])
    
    count_wrong = len(converted_string) - count_correct

    test, train = split_list(converted_string)
    shuffle(train)
    train_0, train_1, train_2, train_3, train_4 = chunks(train, sub_len)
    print("  Test : {}".format(len(test)))
    print("Train0 : {}".format(len(train_0)))
    print("Train1 : {}".format(len(train_1)))
    print("Train2 : {}".format(len(train_2)))
    print("Train3 : {}".format(len(train_3)))
    print("Train4 : {}".format(len(train_4)))

    write(top_path, api, train_0, train_1+train_2+train_3+train_4, 0)
    write(top_path, api, train_1, train_0+train_2+train_3+train_4, 1)
    write(top_path, api, train_2, train_0+train_1+train_3+train_4, 2)
    write(top_path, api, train_3, train_0+train_1+train_2+train_4, 3)
    write(top_path, api, train_4, train_0+train_1+train_2+train_3, 4)



def write_file(top_path, api, valid, train, cross_id):
    with open(os.path.join(top_path, 'train_{}_{}.json'.format(api, cross_id)), 'w') as f:
        f.write('[{}]'.format(','.join(train)))
    with open(os.path.join(top_path, 'valid_{}_{}.json'.format(api, cross_id)), 'w') as f:
        f.write('[{}]'.format(','.join(valid)))
    with open("data_log.txt", "a") as f:
        f.write('{}\t{}\t{}\t{}\n'.format(time.strftime("%Y-%m-%d-%H-%M-%S"), api, count_correct, count_wrong))


if __name__ == '__main__' :
    if len(sys.argv) > 2:
        # pattern = 'java.util.Date.getTime()::java.lang.System.currentTimeMillis()'
        pattern = sys.argv[2]
        top_path = './embed/' + sys.argv[1]
        api = sys.argv[1].replace('/', '_')
        body_api = False
        if len(sys.argv) >= 4:
            body_api = sys.argv[3]
        process(top_path, api, pattern, body_api)
    else :
        with open("patterns.txt", "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) != 0 and not line.startswith("#"):
                    args = line.split()
                    pattern = args[1]
                    top_path = './embed/' + args[0]
                    api = args[0].replace('/', '_')
                    body_api = False
                    if len(args) >= 3:
                        body_api = args[2]

                    process(top_path, api, pattern, body_api)