#include <upc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

shared double *shared *matrixA;
struct timeval stop, start;

void fillMatrixZero(double **matrix, long size){
	long i, j;
	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			matrix[i][j] = 0;
		}
	}
}

void **swapMatrix(long size){
	long i, j;
	double temp;
	for (i = 0; i < size; i++){
		for (j = i + 1; j < size; j++){
			temp = matrixA[i][j];
			matrixA[i][j] = matrixA[j][i];
			matrixA[j][i] = temp;
		}
	}
}

double **createMatrix(long size){
	double **matrix;
	long i;
	matrix = (double **)malloc(size * sizeof(double *));
	for (i = 0; i < size; i++){
		matrix[i] = (double *)malloc(size * sizeof(double));
	}
	return matrix;
}

void printMatrix(double **matrix, long size){
	long i, j;
	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			printf("%2.2f ", matrix[i][j]);
		}
		printf("\n");
	}
}

void printAMatrix(long size){
	long i, j;
	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			printf("%2.2f ", matrixA[i][j]);
		}
		printf("\n");
	}
}

double **L_generate(long size){
	long i, j;
	double **matrixL = createMatrix(size);
	fillMatrixZero(matrixL, size);
	for (i = 0; i < size; i++){
		for (j = 0; j <= i; j++){
			matrixL[i][j] = matrixA[i][j];
			if (i == j){
				matrixL[i][j] = 1.0;
			}
		}
	}
	return matrixL;
}

double **U_generate(long size){
	long i, j;
	double **matrixU = createMatrix(size);
	fillMatrixZero(matrixU, size);
	for (i = 0; i < size; i++){
		for (j = i; j < size; j++){
			matrixU[i][j] = matrixA[i][j];
		}
	}
	return matrixU;
}

double **read_matrix_from_file(int size, char *filename){
	FILE *pf;
	pf = fopen(filename, "r");
	if (pf == NULL)
		return 0;
	int i, j;
	double **matrix = createMatrix(size);
	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			fscanf(pf, "%lf", &(matrix[i][j]));
		}
	}
	fclose(pf);
	return matrix;
}

int main(int argc, char *argv[]){
	long matrixSize;
	double factor, new_value;

	matrixSize = atol(argv[1]);
	char *filename = argv[2];
	double **matrix = read_matrix_from_file(matrixSize, filename);

	matrixA = (shared double *shared *)upc_all_alloc(matrixSize, sizeof(shared double *));
	for (int i = 0; i < matrixSize; i++){
		matrixA[i] = upc_alloc(matrixSize * sizeof(shared double));
	}

	if (MYTHREAD == 0){
		gettimeofday(&start, NULL);
		for (int i = 0; i < matrixSize; i++){
			for (int j = 0; j < matrixSize; ++j){
				upc_memput(&matrixA[i][j], &matrix[j][i], sizeof(double));
			}
		}

		printf("\nMacierz wejsciowa\n");
		printMatrix(matrix, matrixSize);
	}

	upc_barrier;
	upc_forall(int k = 0; k < matrixSize; k++; k){

			for (int j = k + 1; j < matrixSize; j++){
				factor = matrixA[k][j] / matrixA[k][k];
				upc_memput(&matrixA[k][j], &factor, sizeof(double));
			}

			for (int i = k + 1; i < matrixSize; i++){
				for (int j = k + 1; j < matrixSize; j++){
					new_value = matrixA[i][j] - matrixA[i][k] * matrixA[k][j];
					upc_memput(&matrixA[i][j], &new_value, sizeof(double));
				}
			}
			printf("numer iteracji: %d,  numer watku: %d \n", k, MYTHREAD);
	}

	upc_barrier;
	if (MYTHREAD == 0){
		gettimeofday(&stop, NULL);
		double duration = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);

		swapMatrix(matrixSize);
		double **L = L_generate(matrixSize);
		double **U = U_generate(matrixSize);

		printf("Macierz wynikowa:\n");
		printAMatrix(matrixSize);

		printf("L\n");
		printMatrix(L, matrixSize);
		printf("U\n");
		printMatrix(U, matrixSize);
		printf("\n");
		printf("Rozmiar macierzy:\t %lu \n", matrixSize);
		printf("Ilosc watkow:\t\t %d\n", THREADS);
		printf("Czas dzialania:\t %f [s]\n", duration);
	}

	for (int i = 0; i < matrixSize; i++){
		free(matrix[i]);
	}
	free(matrix);

	return 0;
}