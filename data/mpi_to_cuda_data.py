import sys
import os

filename = sys.argv[1]
basefile = filename.split('/')[-1]
V,E=0,0
with open(filename,"r") as f:
	with open("part2","w") as fout:
		while True:
			x = f.readline()
			if len(x) == 0 or x == "\n":
				break
			V +=1
			x = x.split()
			if len(x) <= 1:
				continue
			for i in range(1,len(x)):
				E+=1
				fout.write("%s %s\n"%(x[0],x[i]))
with open("part1","w") as fout:
	fout.write("%s %s 0 0\n"%(V,E))

commmand = "cat part1 part2 > cuda_"+basefile
os.system(commmand)
commmand = "rm -f part1 part2"
os.system(commmand)