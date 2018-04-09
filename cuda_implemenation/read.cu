#include <iostream>
#include <string>
#include <fstream>

using namespace std;

void read(string filename, double** P_sparse, int** row_ind, int** col_ind, int* nnz, int * n)
{
	fstream f(filename.c_str());
	int v, e;
	int ignore;

	f >> v;
	f >> e;
	f >> ignore;
	
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
		f >> (*row_ind)[i];
		(*row_ind)[i]--;
		f >> curRow;
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
