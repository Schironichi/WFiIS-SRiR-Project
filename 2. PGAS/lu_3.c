/*UPC example LU decomposition */
#include <upc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

int maxPrintSize = 5;
int maxTestSize = 100;

shared double *shared *matrixA;
upc_lock_t *l;

struct timeval stop, start;


void readMatrix(double **matrix, long size, FILE* file) {
    long i, j;
    while (!feof (file)){
		for(i = 0; i < size; i++){
			for(j = 0; j < size; j++){
				fscanf(file, "%lf", &(matrix[i][j]));
			}
		}
	}
}

void fillMatrixZero(double **matrix, long size) {
    long i, j;
    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            matrix[i][j] = 0;
        }
    }
}

void ** swapMatrix(long size) {
	long i,j;
	double temp;
	for(i = 0; i < size; i++){
		for(j = i+1; j < size; j++){
			temp = matrixA[i][j];
			matrixA[i][j] = matrixA[j][i];
			matrixA[j][i] = temp;
		}
	}
}

void ** swapMatrix2(double** matrix, long size) {
	long i,j;
	double temp;
	for(i = 0; i < size; i++){
		for(j = i+1; j < size; j++){
			temp = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = temp;
		}
	}
}

double **createMatrix(long size) {
	double **dst; long i;
	dst = (double**) malloc(size * sizeof(double*) );
	for(i = 0; i < size; i++){
		dst[i] = (double*) malloc(size * sizeof(double) );
	}
	return dst;
}

void copyMatrix(double **matrix, double **dst,  long size) {
	long i,j;
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			dst[i][j] = matrix[i][j];
		}
	}
}

void printMatrix(double **matrix, long size) {
	long i,j;
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			printf("%2.5f ", matrix[i][j]);
		}
		printf("\n");
	}
}

void printMainMatrix(long size) {
	long i,j;
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			printf("%2.5f ", matrixA[i][j]);
		}
		printf("\n");
	}
}

double** multiplyMatrixes(double **first, double** second, long size){
	long i,j,k;
	double** solution = createMatrix(size);
	fillMatrixZero(solution, size);
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			for(k = 0; k < size; k++) { 
				solution[i][j] += first[i][k] * second[k][j];
			}
		}
	}
	return solution;
}

double **getL(long size) {
	long i,j;
	double** matrixL = createMatrix(size);
	fillMatrixZero(matrixL, size);
	for(i = 0; i < size; i++){
		for(j = 0; j <= i; j++){
			matrixL[i][j] = matrixA[i][j];
			if(i==j) {
                matrixL[i][j] = 1.0;
			}
		}
	}
	return matrixL;
}

double **getU(long size) {
	long i,j;
	double** matrixU = createMatrix(size);
	fillMatrixZero(matrixU, size);
	for(i = 0; i < size; i++){
		for(j = i; j < size; j++){
			matrixU[i][j] = matrixA[i][j];
		}
	}
	return matrixU;
}

int testResult(double **original, int size){
	long i, j;
	double ** L = getL(size);
	double ** U = getU(size);
	if (size < maxPrintSize) {
		printf("L\n");
		printMatrix(L, size);
		printf("U\n");
		printMatrix(U, size);
	}
	printf("\n");
	
	double** solution = multiplyMatrixes(L, U, size);
	
	if (size < maxPrintSize) {
		printf("LxU\n");
		printMatrix(original, size);
	}
	printf("\n");
	
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (round(original[i][j]) != round(solution[i][j]) ) {
				return 1;
			}
		}
	}
	return 1;
}


int main(int argc, char *argv[]) {

	FILE *file;
	file = fopen("data.txt", "r");
	if (file == NULL){	
   		 printf("Couldn't open input file data.txt!\n");
   		 return -1;
	}
  
	long matrixSize;
	double dzielenie;
	//Czytanie macierzy z pliku
	fscanf(file, "%lu", &matrixSize); 
	double **matrix = createMatrix(matrixSize);
	
	matrixA = (shared double *shared *)upc_all_alloc(matrixSize,sizeof(shared double*));
  	for(int i = 0; i < matrixSize; i++) {
    	matrixA[i] = upc_alloc(matrixSize*sizeof(shared double));
	}

	
    
    if(MYTHREAD == 0) {
		gettimeofday(&start, NULL);
		readMatrix(matrix, matrixSize, file);
	  	for(int i = 0; i < matrixSize; i++) {
			
			for(int j = 0; j < matrixSize; ++j) {
				upc_memput(&matrixA[i][j], &matrix[j][i], sizeof(double)); 
			}
		}

		if(matrixSize < maxPrintSize){
			printf("\n---------- Macierz wejściowa ----------\n");
			printMatrix(matrix, matrixSize);
		}
    }


    upc_barrier 0;
    upc_forall(int k = 0; k < matrixSize; k++; k) {
		
		for(int j = k + 1; j < matrixSize; j++) {  
			dzielenie = matrixA[k][j] / matrixA[k][k];
			upc_memput(&matrixA[k][j], &dzielenie, sizeof(double));
		}           
		
        for(int i = k + 1; i < matrixSize; i++) {
            for(int j = k + 1; j < matrixSize; j++) {
                dzielenie = matrixA[i][j] - matrixA[i][k] * matrixA[k][j];
                upc_memput(&matrixA[i][j], &dzielenie, sizeof(double));
            }   
        }
    }


    upc_barrier 1;
	if (MYTHREAD == 0) {	
	  	gettimeofday(&stop, NULL);
		double sc  = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
        
        swapMatrix(matrixSize);
		printf("\n---------- WYNIK ----------\n");
		if (matrixSize < maxPrintSize) {
			printf("Macierz wynikowa:\n");
			printMainMatrix(matrixSize);
		}
		printf("Wymiar macierzy:\t %lu \n", matrixSize);
		printf("Liczba watkow:\t\t %d\n", THREADS);
		printf("Czas dekompozycji:\t %f [s]\n", sc);
		if (matrixSize < maxTestSize ){
			printf("\n---------- SPRAWDZENIE ----------\n");
			printf("Test poprzez wymnożenie: %s\n", testResult(matrix, matrixSize) ? "Udany" : "Nieudany");
		}
	} 

	for(int i = 0; i < matrixSize; i++){
		free(matrix[i]);
	}
	free(matrix);
	
  	return 0;

} 