// #include "mpi.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <time.h>
#include <upcxx/upcxx.hpp>
constexpr int MAX_PROCESSES = 100;

shared double *shared *matrixA;

using namespace std;

ifstream process_cmd_arguments(int argc, char *argv[]);

double **create_matrix(int numrows, int numcols);

void delete_matrix(double **matrix, int rows);

void print_matrix(double **matrix, int rows);

void load_matrix_and_init_q(ifstream &file, double **matrixA, double **matrixQ, int rows);

//void set_displs_and_send_counts(int x, int y, int size, int *displs, int *send_counts);

int main(int argc, char *argv[]) {
    int rank = upcxx::rank_me();
    int size = upcxx::rank_n();
    int rows, tmpLines, tmpLines2;
    ifstream file;
    double **matrixA, **matrixQ, **mat, **p, **matTmp, **matTmp2;
    int displs[MAX_PROCESSES], send_counts[MAX_PROCESSES], displs2[MAX_PROCESSES], send_counts2[MAX_PROCESSES];
    double *vec;
    double x;
    upcxx::init();
    // UPC++ equivalent of MPI_COMM_WORLD
    upcxx::team team = upcxx::world();

    if (rank == 0) {
        file = process_cmd_arguments(argc, argv);
        file >> rows;
    }

    upcxx::broadcast(&rows, 1, 0).wait();

    matrixA = create_matrix(rows, rows);
    matrixQ = create_matrix(rows, rows);

    if (rank == 0)
        load_matrix_and_init_q(file, matrixA, matrixQ, rows);

    for (int i = 0; i < rows; i++) {
        if (i > 0 && i < rows - size)
            delete_matrix(mat, tmpLines);

        tmpLines = (rows - i) / size;
        if (rank == (size - 1) && size > 1)
            tmpLines += (rows - i) % size;

        mat = create_matrix(tmpLines, rows - i);

        upcxx::broadcast(&matrixQ[0][0], rows * rows, 0).wait();
        upcxx::broadcast(&matrixA[0][0], rows * rows, 0).wait();

        set_displs_and_send_counts(rows - i, rows - i, size, displs, send_counts);
        set_displs_and_send_counts(rows, rows - i, size, displs2, send_counts2);

        x = 0;
        if (i > 0 && i < rows - size) {
            delete[] vec;
            vec = nullptr;
        }

        vec = new double[rows - i];

        if (rank == 0) {
            for (int j = 0; j < rows - i; j++) {
                vec[j] = -matrixA[j + i][i];
                x += vec[j] * vec[j];
            }

            x = sqrt(x);

            if (vec[0] > 0)
                x = -x;

            vec[0] = vec[0] + x;
            x = 0;

            for (int j = 0; j < rows - i; j++) {
                x += vec[j] * vec[j];
            }

            x = sqrt(x);
        }

        upcxx::broadcast(&x, 1, 0).wait();
        upcxx::broadcast(&vec[0], rows - i, 0).wait();

        if (x > 0) {
            if (rank == 0) {
                for (int j = 0; j < rows - i; j++) {
                    vec[j] /= x;
                }
            }

            upcxx::broadcast(&vec[0], rows - i, 0).wait();

            if (i > 0 && i < rows - size)
                delete_matrix(p, rows - i);

            p = create_matrix(rows - i, rows - i);

            // upcxx::scatter(&p[0][0], send_counts, displs, 0).wait();
            upcxx::scatter_into(p ? &p[0][0] : nullptr, send_counts, displs, matTmp, send_counts[rank], 0, team);

            for (int k = 0; k < send_counts[rank] / (rows - i); k++) {
                for (int l = 0; l < rows - i; l++) {
                    if ((k + (displs[rank] / (rows - i))) == l)
                        mat[k][l] = 1 - 2 * vec[k + displs[rank] / (rows - i)] * vec[l];
                    else
                        mat[k][l] = -2 * vec[k + displs[rank] / (rows - i)] * vec[l];
                }
            }

            upcxx::gather_into(mat[0], send_counts[rank], p ? &p[0][0] : nullptr, send_counts, displs, 0, team).wait();
            upcxx::broadcast(&p[0][0], (rows - i) * (rows - i), 0).wait();

            for (int k = 0; k < send_counts[rank] / (rows - i); k++) {
                for (int l = 0; l < rows - i; l++) {
                    double tm = 0;
                    for (int m = i; m < rows; m++) {
                        tm += p[k + displs[rank] / (rows - i)][m - i] * matrixA[m][l + i];
                    }
                    mat[k][l] = tm;
                }
            }

            if (i > 0 && i < rows - size)
                delete_matrix(matTmp, rows - i);

            matTmp = create_matrix(rows - i, rows - i);

            upcxx::gather_into(mat[0], send_counts[rank], matTmp ? &matTmp[0][0] : nullptr, send_counts, displs, 0, team).wait();

            if (rank == 0) {
                for (int k = i; k < rows; k++) {
                    for (int l = i; l < rows; l++) {
                        matrixA[k][l] = matTmp[k - i][l - i];
                    }
                }
            }

            if (i > 0 && i < rows - size)
                delete_matrix(mat, tmpLines2);

            tmpLines2 = (rows) / size;

            if (rank == (size - 1) && size > 1)
                tmpLines2 += (rows) % size;

            mat = create_matrix(tmpLines2, rows - i);

            for (int k = 0; k < send_counts2[rank] / (rows - i); k++) {
                for (int l = 0; l < rows - i; l++) {
                    double tm = 0;
                    for (int m = i; m < rows; m++) {
                        tm += matrixQ[k + displs2[rank] / (rows - i)][m] * p[m - i][l];
                    }
                    mat[k][l] = tm;
                }
            }

            if (i > 0 && i < rows - size)
                delete_matrix(matTmp, rows);

            matTmp = create_matrix(rows, rows - i);

            upcxx::gather(&mat[0][0], send_counts2[rank], &matTmp[0][0], send_counts2, displs2, 0).wait();

            if (rank == 0) {
                for (int k = 0; k < rows; k++) {
                    for (int l = i; l < rows; l++) {
                        matrixQ[k][l] = matTmp[k][l - i];
                    }
                }
            }
        }
    }

    if (rank == 0) {
        cout << "\nMatrix Q:\n";
        print_matrix(matrixQ, rows);
        cout << "\nMatrix R:\n";
        print_matrix(matrixA, rows);
        cout.precision(6);
        // cout << "\nExecution time: " << (double)(clock() - tStart) / CLOCKS_PER_SEC << "s\n";
    }

    file.close();
    upcxx::finalize();
    return 0;
}


/**
 * Process command line arguments.
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return An input file stream object
 * @throws An error and exits the program if no input file or more than one input file is provided or if the input file cannot be opened
 */
ifstream process_cmd_arguments(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Please provide input file as a single argument." << endl;
        upcxx::finalize();
        exit(1);
    }
    else
    {
        string filename = argv[1];
        ifstream file(filename);
        if (file.is_open())
            return file;
        else
        {
            cout << "Cannot open file." << endl;
            upcxx::finalize();
            exit(1);
        }
    }
}
/**
 * Creates a dynamic two-dimensional matrix of doubles.
 * @param numrows: an integer indicating the number of rows in the matrix to be created.
 * @param numcols: an integer indicating the number of columns in the matrix to be created.
 * @return a pointer to the matrix.
 */
double **create_matrix(int numrows, int numcols)
{
    double *buffer = new double[numrows * numcols]; // allocate a contiguous block of memory to store the matrix
    double **data = new double *[numrows];          // allocate an array of pointers to store the starting addresses of the rows

    for (int i = 0; i < numrows; ++i)
        data[i] = buffer + i * numcols; // set the pointers in the data array to point to the start of each row in the buffer

    return *&data; // return a pointer to the matrix
}

/**
 * Deletes the matrix with a specified number of rows
 *
 * @param matrix The matrix to be deleted
 * @param rows The number of rows in the matrix
 */
void delete_matrix(double **matrix, int rows)
{
    delete[] matrix[0];
    matrix = nullptr;
}

/**
 * Prints a matrix with a specified number of rows
 *
 * @param matrix The matrix to be printed
 * @param rows The number of rows in the matrix
 */
void print_matrix(double **matrix, int rows)
{
    cout.precision(3);             // Set the precision of the output
    for (int i = 0; i < rows; i++) // Iterate over each row in the matrix
    {
        for (int j = 0; j < rows; j++)             // Iterate over each column in the row
            cout << matrix[i][j] << fixed << "\t"; // Print the element with a fixed precision and a tab separator
        cout << "\n";                              // Print a newline character to separate the rows
    }
}

/**
 * Loads matrix A from file and initializes matrix Q to the identity matrix.
 *
 * @param file The input file stream to load matrix A from.
 * @param matrixA The matrix A to be loaded.
 * @param matrixQ The matrix Q to be initialized to the identity matrix.
 * @param rows The number of rows (and columns) in the matrix A and Q.
 */
void load_matrix_and_init_q(ifstream &file, double **matrixA, double **matrixQ, int rows)
{
    // Loop through each element of the matrix and load it from file into matrixA
    // Set diagonal elements of matrixQ to 1 and other elements to 0
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            file >> matrixA[i][j];

            matrixQ[i][j] = 0;
            if (i == j)
                matrixQ[i][j] = 1;
        }
    }
    // Print the loaded matrix A to the console
    cout << "\nMatrix loaded from file:\n";
    print_matrix(matrixA, rows);
}
/**
 * Sets the displacement and send counts arrays
 * @param x: number of rows in matrix
 * @param y: number of columns in matrix
 * @param size: number of MPI processes
 * @param displs: array of displacements (output parameter)
 * @param send_counts: array of send counts (output parameter)
 */
void set_displs_and_send_counts(int x, int y, int size, int *displs, int *send_counts)
{
    int part = (x / size) * y; // calculate the part size of the matrix to be sent to each process
    send_counts[0] = part;     // set the send count for process 0
    displs[0] = 0;             // set the displacement for process 0
    if (size > 1)              // if there are more than one processes
    {
        for (int i = 1; i < size - 1; i++) // iterate from process 1 to process size - 2
        {
            send_counts[i] = part;            // set the send count for the current process
            displs[i] = displs[i - 1] + part; // calculate the displacement for the current process
        }
        displs[size - 1] = displs[size - 2] + part;           // set the displacement for the last process
        send_counts[size - 1] = ((x * y) - displs[size - 1]); // set the send count for the last process
    }
}
