#include <upcxx/upcxx.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <time.h>

constexpr int MAX_PROCESSES = 100;

using namespace std;

ifstream process_cmd_arguments(int argc, char *argv[]);

double **create_matrix(int numrows, int numcols);

void delete_matrix(double **matrix, int rows);

void print_matrix(double **matrix, int rows);

void load_matrix_and_init_q(ifstream &file, double **matrixA, double **matrixQ, int rows);

void scatter(double *source, int srows, int scols, int rank, double *target, int recvstart, int count);

void gather(double *source, int rank, double *target, int trows, int tcols, int sendstart, int count);

void set_displs_and_send_counts(int x, int y, int size, int *displs, int *send_counts);

int main(int argc, char *argv[])
{
    int rank = 0, size, rows, tmpLines, tmpLines2;
    ifstream file;
    double **matrixA, **matrixQ, **mat, **p, **matTmp, **matTmp2;
    double *vec;
    double x;
    int displs[MAX_PROCESSES], send_counts[MAX_PROCESSES], displs2[MAX_PROCESSES], send_counts2[MAX_PROCESSES];
    clock_t tStart;

    // initialize UPC++
    upcxx::init();
    // get current process rank
    rank = upcxx::rank_me();
    // get total number of processes
    size = upcxx::rank_n();

    if (rank == 0)
    {
        // get input file from command line arguments and exit if not provided or cannot be opened.
        file = process_cmd_arguments(argc, argv);

        // scanning file for number of rows
        file >> rows;

        tStart = clock();
    }
    // broadcast the total number of rows to all processes
    upcxx::broadcast(&rows, 1, 0).wait();
    // create matrices to store the A and Q matrices
    matrixA = create_matrix(rows, rows);
    matrixQ = create_matrix(rows, rows);

    if (rank == 0)
        // Loading matrix A from file and initialization matrix Q to the identity matrix.
        load_matrix_and_init_q(file, matrixA, matrixQ, rows);

    for (int i = 0; i < rows; i++)
    {
        if (i > 0 && i < rows - size)
            delete_matrix(mat, tmpLines);
        // calculate number of rows to be processed by the current process    
        tmpLines = (rows - i) / size;
        if (rank == (size - 1) && size > 1)
            tmpLines += (rows - i) % size;

        // create temporary matrix for parallel computation
        mat = create_matrix(tmpLines, rows - i);
        // broadcast the A and Q matrices to all processes
        upcxx::broadcast(&matrixQ[0][0], rows * rows, 0).wait();
        upcxx::broadcast(&matrixA[0][0], rows * rows, 0).wait();

        // set displacement and send counts for the matrix of size (rows-i)x(rows-i)
        set_displs_and_send_counts(rows - i, rows - i, size, displs, send_counts);
        // set displacement and send counts for the matrix of size rows x (rows-i)
        set_displs_and_send_counts(rows, rows - i, size, displs2, send_counts2);

        x = 0;
        if (i > 0 && i < rows - size)
        {
            delete[] vec;
            vec = nullptr;
        }
        vec = new double[rows - i];
        if (rank == 0)
        {
            // calculate the norm of the current column of A matrix
            for (int j = 0; j < rows - i; j++)
            {
                vec[j] = -matrixA[j + i][i];
                x += vec[j] * vec[j];
            }

            x = sqrt(x);

            if (vec[0] > 0)
                x = -x;
            vec[0] = vec[0] + x;
            x = 0;
            for (int j = 0; j < rows - i; j++)
            {
                x += vec[j] * vec[j];
            }
            x = sqrt(x);
        }
        // Broadcast x and vec[0] to all processes.
        upcxx::broadcast(&x, 1, 0).wait();
        upcxx::broadcast(&vec[0], rows - i, 0).wait();
        // If x is greater than 0, normalize vec.
        if (x > 0)
        {
            if (rank == 0)
            {
                // normalize vec
                for (int j = 0; j < rows - i; j++)
                {
                    vec[j] /= x;
                }
            }
            // Broadcast the normalized vec to all processes.
            upcxx::broadcast(&vec[0], rows - i, 0).wait();

            // Construct the matrix P.
            // If i > 0 and i < rows - size, delete the previous version of P.
            if (i > 0 && i < rows - size)
                delete_matrix(p, rows - i);
            p = create_matrix(rows - i, rows - i);
            // Scatter the rows of P to all processes.
            scatter(&p[0][0], rows - i, rows - i, rank, &mat[0][0], displs[rank], tmpLines * (rows - i));
            upcxx::barrier();
            // Compute the values of P for the assigned rows of mat.
            for (int k = 0; k < send_counts[rank] / (rows - i); k++)
            {
                for (int l = 0; l < rows - i; l++)
                {
                    if ((k + (displs[rank] / (rows - i))) == l)
                        mat[k][l] = 1 - 2 * vec[k + displs[rank] / (rows - i)] * vec[l];
                    else
                        mat[k][l] = -2 * vec[k + displs[rank] / (rows - i)] * vec[l];
                }
            }
            // Gather the rows of P computed by each process to the root process.
            gather(&mat[0][0], rank, &p[0][0], rows - i, rows - i, displs[rank], send_counts[rank]);
            upcxx::barrier();
            // Broadcast the entire matrix P to all processes.
            upcxx::broadcast(&p[0][0], (rows - i) * (rows - i), 0).wait();

            // Multiply the assigned rows of matrix A by the assigned rows of P.
            // Store the result in mat. (parellel)
            for (int k = 0; k < send_counts[rank] / (rows - i); k++)
            {
                for (int l = 0; l < rows - i; l++)
                {
                    double tm = 0;
                    for (int m = i; m < rows; m++)
                    {
                        tm += p[k + displs[rank] / (rows - i)][m - i] * matrixA[m][l + i];
                    }
                    mat[k][l] = tm;
                }
            }
            // If i > 0 and i < rows - size, delete the previous version of matTmp.
            if (i > 0 && i < rows - size)
                delete_matrix(matTmp, rows - i);

            // Create a temporary matrix to hold data from a portion of the original matrix
            // to be processed by each process.    
            matTmp = create_matrix(rows - i, rows - i);
            // Gather data from the original matrix into matTmp, using send_counts and displs
            // arrays to specify the portion of data to be sent by each process.
            gather(&mat[0][0], rank, &matTmp[0][0], rows - i, rows - i, displs[rank], send_counts[rank]);
            upcxx::barrier();
            // If the current process is the root process (rank 0), copy the data from matTmp back
            // into the appropriate portion of the original matrix.
            if (rank == 0)
            {
                for (int k = i; k < rows; k++)
                {
                    for (int l = i; l < rows; l++)
                    {
                        matrixA[k][l] = matTmp[k - i][l - i];
                    }
                }
            }

            // Delete the memory allocated for mat and tmpLines2 if this is not the first iteration
            // and if i is not equal to rows - size.
            if (i > 0 && i < rows - size)
                delete_matrix(mat, tmpLines2);

            // Calculate the number of rows of the temporary matrix to be created by each process.
            tmpLines2 = (rows) / size;
            // If the current process is the last process and there is a remainder after dividing
            // the number of rows by the number of processes, add this remainder to the number of
            // rows to be processed by the current process.
            if (rank == (size - 1) && size > 1)
                tmpLines2 += (rows) % size;
            // Create a new temporary matrix to hold intermediate results for the calculation of the Q matrix.
            mat = create_matrix(tmpLines2, rows - i);

            // Loop through the data received by the current process and calculate the corresponding portion of the Q matrix.
            for (int k = 0; k < send_counts2[rank] / (rows - i); k++)
            {
                for (int l = 0; l < rows - i; l++)
                {
                    double tm = 0;
                    for (int m = i; m < rows; m++)
                    {
                        tm += matrixQ[k + displs2[rank] / (rows - i)][m] * p[m - i][l];
                    }
                    mat[k][l] = tm;
                }
            }
            // Delete the memory allocated for matTmp if this is not the first iteration
            // and if i is not equal to rows - size.
            if (i > 0 && i < rows - size)
                delete_matrix(matTmp, rows);
            // Create a new temporary matrix to hold the final results for the Q matrix.
            matTmp = create_matrix(rows, rows - i);
            // Gather data from the temporary matrix into matTmp, using send_counts2 and displs2
            // arrays to specify the portion of data to be sent by each process.
            gather(&mat[0][0], rank, &matTmp[0][0], rows, rows - i, displs2[rank], send_counts2[rank]);
            upcxx::barrier();
            
            // If this process has rank 0, then loop copies the data from matTmp back to the original matrixQ
            // by offsetting the column index by i.
            if (rank == 0)
            {

                for (int k = 0; k < rows; k++)
                {
                    for (int l = i; l < rows; l++)
                    {
                        matrixQ[k][l] = matTmp[k][l - i];
                    }
                }
            }
        }
    }
    // If this process has rank 0, then print the resulting matrixQ and matrixA
    if (rank == 0)
    {
        cout << "\nMatrix Q:\n";
        print_matrix(matrixQ, rows);
        cout << "\nMatrix R:\n";
        print_matrix(matrixA, rows);
        cout.precision(6);  
        cout << "\nExecution time: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << "s\n";
    }
    // Close the input file and finalize UPC++
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
 * @param size: number of UPC++ processes
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

/**
 * Scatter function scatters data from the source array to the target array using UPC++.
 *
 * @param source The pointer to the source array.
 * @param srows The number of rows in the source array.
 * @param scols The number of columns in the source array.
 * @param rank The rank of the current process.
 * @param target The pointer to the target array.
 * @param recvstart The starting index in the target array where data should be received.
 * @param count The number of elements to be scattered from the source array to the target array.
 */
void scatter(double *source, int srows, int scols, int rank, double *target, int recvstart, int count)
{
    // Declare a global pointer to store the distributed data
    upcxx::global_ptr<double> matrix;

    if (rank == 0)
    {
        // Allocate a new array to store the data on rank 0
        matrix = upcxx::new_array<double>(srows * scols);
        
        // Get the local pointer to the matrix data
        double *local_matrix = matrix.local();

        // Copy the data from the source array to the local_matrix
        for (int i = 0; i < srows * scols; i++) {
            local_matrix[i] = *(source + i);
        }
    }

    // Broadcast the matrix pointer to all ranks
    matrix = upcxx::broadcast(matrix, 0).wait();

    // Scatter the data from the source array to the target array
    for (int i = 0; i < count; i++)
        *(target + i) = rank ? upcxx::rget(matrix + recvstart + i).wait() : *(source + (recvstart + i));
}


/**
 * @brief Gather function gathers data from the source array and stores it in the target array using UPC++.
 *
 * @param source The pointer to the source array.
 * @param rank The rank of the current process.
 * @param target The pointer to the target array.
 * @param trows The number of rows in the target array.
 * @param tcols The number of columns in the target array.
 * @param sendstart The starting index in the target array where data should be sent.
 * @param count The number of elements to be gathered from the source array to the target array.
 */
void gather(double *source, int rank, double *target, int trows, int tcols, int sendstart, int count)
{
    // Declare a global pointer to store the distributed data
    upcxx::global_ptr<double> matrix;

    if (rank == 0)
    {
        // Allocate a new array on rank 0 to store the gathered data
        matrix = upcxx::new_array<double>(trows * tcols);
    }

    // Broadcast the matrix pointer to all ranks
    matrix = upcxx::broadcast(matrix, 0).wait();

    if (rank == 0)
    {
        // Copy data from the source array to the target array on rank 0
        for (int i = 0; i < count; i++)
            *(target + (sendstart + i)) = *(source + i);
    }
    else
    {
        // Put data from the source array into the matrix on other ranks
        for (int i = 0; i < count; i++)
            upcxx::rput(*(source + i), matrix + sendstart + i).wait();
    }

    // Synchronize all ranks
    upcxx::barrier();

    if (rank == 0)
    {
        // Get the local pointer to the matrix data on rank 0
        double *local_matrix = matrix.local();

        // Copy the remaining data from the matrix to the target array on rank 0
        for (int i = count; i < trows * tcols; i++)
            *(target + (sendstart + i)) = local_matrix[i];
    }
}
