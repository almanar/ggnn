#!/usr/bin/env python
# coding=utf-8

import io
import os
import csv

# train_APIReplacement_Mkdir_0.json_result_valid.txt

def parse(f, dic):
	with open(f, 'r') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		next(csv_reader)
		for row in csv_reader:
			epoch = int(row[0])
			# fname = row[1]
			loss = float(row[2])
			append(dic['loss'], epoch, loss)

			accs = float(row[3])
			append(dic['acc'], epoch, accs)

			prec = float(row[4])
			append(dic['prec'], epoch, prec)

			recall = float(row[5])
			append(dic['recall'], epoch, recall)

			f1 = float(row[6])
			append(dic['f1'], epoch, recall)

			speed = float(row[7])
			append(dic['speed'], epoch, speed)
			

def append(dic, key, value):
	if key not in dic:
		dic[key] = [value]
	else :
		dic[key].append(value)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Please input API path like "APIReplacement/Mkdir"')
	else:
		dic = {}
		dic['loss'] = {}
		dic['acc'] = {}
		dic['prec'] = {}
		dic['recall'] = {}
		dic['f1'] = {}
		dic['speed'] = {}

		api = sys.argv[1].replace('/', '_')
		base = os.path.join("./logs", api)
		for i in range(5):
			f = os.path.join(base, 'train_{}_{}.json_result_valid.txt'.format(api, i))
			parse(f, dic)
	for key, value in dic['acc']:
		print(key, value)
