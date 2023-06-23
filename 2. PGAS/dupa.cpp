#include <iostream>
#include <upcxx/upcxx.hpp>

int main(int argc, char argv[]) {
    upcxx::init();

    int rows = 5; // Liczba wierszy macierzy
    int cols = 4; // Liczba kolumn macierzy

    // Rank 0 tworzy globalną tablicę i wypełnia ją losowymi liczbami
    upcxx::global_ptr<int> matrix;
    if (upcxx::rank_me() == 0) {
        matrix = upcxx::new_array<int>(rows cols);
        int local_matrix = matrix.local();
        for (int i = 0; i < rows cols; i++) {
            local_matrix[i] = rand() % 100; // Losowe liczby od 0 do 99
        }
    }

    // Rozgłaszanie wskaźnika globalnego do wszystkich procesów
    matrix = upcxx::broadcast(matrix, 0).wait();

    // Każdy rank inny niż 0 wypisuje pojedynczy wiersz macierzy
    if (upcxx::rank_me() != 0) {
        int row_idx = upcxx::rank_me() - 1; // Indeks wiersza do wyświetlenia
        for (int col_idx = 0; col_idx < cols; col_idx++) {
            int elem = upcxx::rget(matrix + row_idx * cols + col_idx).wait();
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    // Rank 0 wypisuje całą macierz
    if (upcxx::rank_me() == 0) {
        int local_matrix = matrix.local();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << local_matrix[i cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    upcxx::finalize();
    return 0;
}
//gather
int main2(int argc, char argv[]) {
    upcxx::init();

    int rows = 5; // Liczba wierszy macierzy
    int cols = 4; // Liczba kolumn macierzy
    int rank_value = upcxx::rank_me(); // Wartość ranku

    // Rank 0 tworzy globalną tablicę
    upcxx::global_ptr<int> matrix;
    if (upcxx::rank_me() == 0) {
        matrix = upcxx::new_array<int>(rows cols);
    }

    // Rozgłaszanie wskaźnika globalnego do wszystkich procesów
    matrix = upcxx::broadcast(matrix, 0).wait();

    // Inne ranki uzupełniają pojedyncze wiersze macierzy swoją wartością
    if (upcxx::rank_me() != 0) {
        int row_idx = upcxx::rank_me() - 1; // Indeks wiersza do uzupełnienia
        for (int col_idx = 0; col_idx < cols; col_idx++) {
            upcxx::rput(rank_value, matrix + row_idx * cols + col_idx).wait();
        }
    }

    // Barrier, aby upewnić się, że wszystkie ranki ukończyły swoje operacje
    upcxx::barrier();

    // Rank 0 wypisuje całą macierz
    if (upcxx::rank_me() == 0) {
        int local_matrix = matrix.local();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << local_matrix[i cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    upcxx::finalize();
    return 0;
}