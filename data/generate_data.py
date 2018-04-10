from random import randint
import sys

if len(sys.argv) != 4:
	print("Usage: python %s number_of_vertices number_of_edges output_filename"%(sys.argv[0]))
	exit(1)
V = int(sys.argv[1])
E = int(sys.argv[2])
filename = sys.argv[3]

with open(filename,"w") as f:
	f.write("%s %s %s\n"%(V,E,1))
	for i in range(1, V + 1):
		if i %100000 == 0 :
			print (i)
		f.write("%s %s\n"%(i,"link"+str(i)))
	for i in range(E):
		if i %100000 == 0 :
			print (i)
		x,y = randint(1,V),randint(1,V)
		f.write("%s %s\n"%(x,y))