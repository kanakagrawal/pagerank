import sys

filename = sys.argv[1]
adj_list = []
with open(filename,"r") as f:
	x= f.readline()
	V,E,temp,index = x.split()
	V = int(V)
	E = int(E)
	temp = int(temp)
	index = int(index)
	for i in range(V):
		adj_list.append([])
	if temp == 1:
		for i in range(V):
			x = f.readline()
	for i in range(E):
		x = f.readline()
		a,b = x.split()
		a = int(a)
		b = int(b)
		if index == 0:
			adj_list[a].append(b)
		else:
			adj_list[a-1].append(b-1)
for i in range(V):
	adj_list[i].sort()
with open("mpi_"+filename,"w") as f:
	for i in range(V):
		x = ' '.join(map(str, adj_list[i]))
		out = str(i) + " " + x + "\n"
		f.write(out)