#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

void read(string filename, double** P_sparse, int** row_ind, int** col_ind, int* nnz, int * n)
{
	fstream f(filename.c_str());
	int v, e;
	int ignore = 0; // Generally no links mapping
	int one_starting = 1;
	std::string::size_type sz;   // alias of size_t


	/* 
	 * Parsing the metadata of data
	 */
	string metadata;
	getline (f, metadata);
	
	vector<string> tokens = split(metadata, ' ');

	v = stoi ( tokens[0], &sz );
	e = stoi ( tokens[1], &sz );
	
	if (tokens.size() == 2) { // Ignore the link to number mapping
		ignore = 1;
	}
	else if (tokens.size() == 3) {
		ignore = stoi ( tokens[2], &sz);
	}
	else if (tokens.size() == 4) {
		ignore = stoi ( tokens[2], &sz);
		one_starting = stoi ( tokens[3], &sz ); 
	}
	
	*nnz = e;
	*n = v;
	if (ignore) {
		string dummy;
		int du;
		for(int i = 0; i < v; i++)
		{
			f >> du >> dummy;
		}
	}
	
	*row_ind = new int[e];
	*col_ind = new int[v + 1];
	
	for(int i = 0; i < v + 1; i++)
	{
		(*col_ind)[i] = 0;
	}
	

	*P_sparse = new double[e];
	
	int curLengthCumulative = 0;
	int curRow, prevRow = 0;
	for(int i = 0; i < *nnz; i++)
	{
		f >> curRow;
		f >> (*row_ind)[i];
		if (one_starting)
			(*row_ind)[i]--;
		if (!one_starting)
			curRow ++;
//		curRow--;
		if (curRow != prevRow)
		{
			for (int j = prevRow + 1; j < curRow; j++)
				(*col_ind)[j] = curLengthCumulative;
			(*col_ind)[prevRow] = curLengthCumulative;	
			prevRow = curRow;
		}
		curLengthCumulative++;
		(*P_sparse)[i] = 1.0;
	}
	(*col_ind)[curRow] = curLengthCumulative;
}
/*
int main()
{
	string filename = "hollins.dat";
	int *row_ind, *col_ind, *nnzPerVectorA;
	double* P_sparse;
	int nnz, n;
	read(filename, &P_sparse, &row_ind, &col_ind, &nnz, &n, &nnzPerVectorA);
	cout << n << endl << nnz << endl;
	
	cout << col_ind[0] << endl << col_ind[n-1] << endl;
}
*/
