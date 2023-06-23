#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

void matrix_print (float *M, size_t dim, size_t sep);
void gauss_elim(float **new_row, float *row, size_t dim);
float *U_generate (float *M, int dim);
float *L_generate (float *M, int dim);
float *read_matrix_from_file(int size, char *filename);
void write_matrix_to_file(float *M, int size, char *filename);

int main(int argc, char *argv[])
{
   srand(time(NULL));
   const int master = 0;
   int size = 0, numprocs, myid;
   if (argc < 2) {
      printf("Matrix size missing in the arguments\n");
      return EXIT_FAILURE;
   }
   size = atol(argv[1]);
   char *filename = argv[2];
   float *A = read_matrix_from_file(size, filename);

   MPI_Init(&argc,&argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // jesli proces jest masterem wypisana jest oryginalna macierz
   if (myid == master) {

      printf("[A]\n");
      matrix_print(A, size * size, size);
      printf("\n");
    
   }

   int i, j, col_num = 0;
   double start = MPI_Wtime();

   // przechodzimy po (size - 1) wierszach 
   for (i = 0; i < size - 1; i++) {
      // do zmiennej row zapisujemy aktualny wiersz
      float *row = &A[i * size + col_num];
      // przechodzimy po kolejnych wierszach
      for (j = i + 1; j < size; j++) {
         // do kazdego wiersza przypisujemy proces, ktory wykona na nim eliminacje Gaussa 
         if (j % numprocs == myid) {
            float *new_row = &A[j * size + col_num];
            gauss_elim(&new_row, row, size - col_num);
         }
      }

      // kazdy nowy wiersz jest odbierany lub wysylany do pozostalych procesow
      for (j = i + 1; j < size; j++) {
         float *new_row = &A[j * size + col_num];
         MPI_Bcast(new_row, size - col_num, MPI_FLOAT, j % numprocs, MPI_COMM_WORLD);
      }

      // czekamy az wszystkie procesy zakoncza prace
      MPI_Barrier(MPI_COMM_WORLD);
      col_num++;
   }

   double end = MPI_Wtime();

   float *L = L_generate(A, size);
   float *U = U_generate(A, size);

   if (myid == master) {

	  printf("\n[LU]\n");
		matrix_print(A, size * size, size);
      write_matrix_to_file(A, size, "output.txt");
      printf("\n[L]\n");
      matrix_print(L, size * size, size);
      write_matrix_to_file(L, size, "output.txt");
      printf("\n[U]\n");
      matrix_print(U, size * size, size);
      write_matrix_to_file(U, size, "output.txt");

      printf("mpi: %f s\n", end - start);
   }
   free(A);


   MPI_Finalize();
   return EXIT_SUCCESS;
}

// generowanie macierzy
float *gen_mx (size_t dim)
{
   int i, j, tot = dim * dim;
   float *M = malloc(sizeof(float) * tot);
   for (i = 0; i < tot; i++) {
      M[i] = rand() % 101 - 50;
   }

   return M;
}

// eliminacja Gaussa
void gauss_elim(float **new_row, float *row, size_t dim)
{
   if (**new_row == 0)
      return;

   // wyznaczamy wspolczynnik k
   float k = **new_row / row[0];

   // wyliczamy wartosci nowego wiersza
   for (int i = 1; i < dim; i++) {
      (*new_row)[i] = (*new_row)[i] - k * row[i];
   }
   **new_row = k;
}

// funkcja wypisujaca macierz
void matrix_print (float *M, size_t dim, size_t sep)
{
   int i, j;
   for (i = 0; i < dim; i++) {
      printf("% 4.2f\t", M[i]);
      if ((i + 1) % sep == 0) {
         printf("\n");
      }
   }
}

// funkcja generujaca macierz U na podstawie macierzy M
float *U_generate (float *M, int dim)
{
   int i, j;
   float *U = malloc(sizeof(float) * dim * dim);
   for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
         if (j >= i) {
            U[i * dim + j] =  M[i * dim + j];
         } else {
            U[i * dim + j] =  0;
         }
      }
   }
   return U;
}

// funkcja generujaca macierz L na podstawie macierzy M
float *L_generate (float *M, int dim)
{
   int i, j;
   float *L = malloc(sizeof(float) * dim * dim);
   for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
         if (j > i) {
            L[i * dim + j] = 0;
         } else if (i == j) {
             L[i * dim + j] = 1;
         } else {
            L[i * dim + j] = M[i * dim + j];
         }
      }
   }
   return L;
}

// funkcja wczytujaca macierz z pliku
float *read_matrix_from_file(int size, char *filename){

    FILE *pf;
    pf = fopen (filename, "r");
    if (pf == NULL)
        return 0;
    int i;
    int matrix_size = size*size;
    float *M = malloc(sizeof(float) * matrix_size);
    for (i = 0; i < matrix_size; i++)
    {
        fscanf(pf, "%f", &M[i]);
    }
    fclose (pf);
    return M;

}

// funkcja zapisujaca macierz do pliku
void write_matrix_to_file(float *M, int size, char *filename){

    FILE *out_file = fopen(filename, "a");
          if (out_file == NULL)
            {  
              printf("Error! Could not open file\n");
              exit(-1); 
            }
            for(int i =0; i< size * size; i++){
                    fprintf(out_file, "%f ", M[i]);
                    if(i%size==size-1){
                        fprintf(out_file, "\n");
                    }
            }
            fprintf(out_file, "\n");
            fclose(out_file);
}