import io
import sys

with open("run_all.sh", "w") as script:
	script.write("#!/bin/bash\n\n")
	with open("data/patterns.txt", "r") as f:
		for line in f.readlines():
			line = line.rstrip()
			if len(line) == 0 or line.startswith("#"):
				script.write("\n")
			else :
				s = line.split()
				path = s[0]
				for i in range(5):
					script.write("python3 ggnn_sparse.py --data_dir data/embed/{} --valid_file valid_{}_{}.json --train_file train_{}_{}.json\n".format(path, path.replace('/', '_'), i, path.replace('/', '_'), i))
				script.write("\n\n")
