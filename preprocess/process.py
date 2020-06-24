import sys
import os
import io
import csv
import json
import ast


def filter(feature_file_name, pattern):
    filtered = []
    csv.field_size_limit(sys.maxsize)
    with open(feature_file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        header = next(csv_reader, None)
        line_count = 0
        # tokens\ttags\tRepAPI\tRetAPI\tBodyAPI\tfile\tline
        for row in csv_reader:
            seq = row[0] # json string for gnn tokens
            tags = set(row[1].split(',')) # separated with ',' when there are more than one
            repAPI = row[2] # with the form of API1::API2
            retAPI = set(row[3].split(',')) # separated with ',' when there are more than one
            bodyAPI = row[4] # separated with ',' when there are more than one
            fname = row[5] # absolute file name of the pattern
            line = int(row[6]) # line number

            if repAPI == pattern:
                filtered.append(row)
                line_count += 1

        print(line_count)
    return filtered

def loadEmbeddingMap():
    dic = {}
    dic['kind'] = {'UNKTYPE': 0}
    dic['op'] = {'UNK': 1}
    dic['type'] = {'Type': 2}
    return dic

def embeddingSingle(dic, feature, defaultValue) :
    for key, value in dic.items():
        print(key, value)
    print(feature)
    if feature in dic :
        return dic[feature]
    else :
        return dic[defaultValue]


default_value_map = {'kind': 'UNKTYPE', 'op': 'UNK', 'type': 'Type'}

def embedding(features):
    dic = loadEmbeddingMap()
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
                    n_kind = embeddingSingle(dic['kind'], item[0], default_value_map['kind'])
                    n_op = embeddingSingle(dic['op'], item[1], default_value_map['op'])
                    n_type = embeddingSingle(dic['type'], item[2], default_value_map['type'])
                    n_api = value[3]
                    if n_api != '' :
                        n_api = 1 # TODO: update
                    items.append([n_kind, n_op, n_type, n_api])
                convertedFeature[key] = items
            else :
                convertedFeature[key] = value
        # convertedFeature['API'] = row[2]
        converted.append(convertedFeature)
    return converted

if __name__ == '__main__' :
    pattern = 'java.util.Date.getTime()::java.lang.System.currentTimeMillis()'
    correct_filtered = filter('correct.txt', pattern)
    correct_converted = embedding(correct_filtered)
    correct_converted_string = [str(ast.literal_eval(json.dumps(item))).replace("'", "\"") for item in correct_converted]
    
    wrong_filtered = filter('wrong.txt', pattern)
    wrong_converted = embedding(wrong_filtered)
    wrong_converted_string = [str(ast.literal_eval(json.dumps(item))).replace("'", "\"") for item in wrong_converted]

    with open('all.json', 'w+') as f:
        f.write('[{},{}]'.format(','.join(correct_converted_string), ','.join(wrong_converted_string)))