#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 5  // Размер сетки
#define MAX_ITER 1000  // Максимальное количество итераций
#define TOLERANCE 1e-6  // Точность

// Функция для инициализации сетки
void initialize_grid(double **grid, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            grid[i][j] = 0.0;
        }
    }
}

// Функция для установки граничных условий
void set_boundary_conditions(double **grid, int n) {
    for (int i = 0; i < n; i++) {
        grid[i][0] = 100.0;  // Левая граница
        grid[i][n-1] = 100.0;  // Правая граница
    }
    for (int j = 0; j < n; j++) {
        grid[0][j] = 0.0;  // Верхняя граница
        grid[n-1][j] = 0.0;  // Нижняя граница
    }
}

// Функция для вычисления нормы разности
double compute_norm(double **grid, double **new_grid, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            norm += (grid[i][j] - new_grid[i][j]) * (grid[i][j] - new_grid[i][j]);
        }
    }
    return sqrt(norm);
}

// Функция для метода Гаусса-Зейделя
void gauss_seidel(double **grid, int n, int max_iter, double tolerance) {
    double **new_grid = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        new_grid[i] = (double *)malloc(n * sizeof(double));
    }

    int iter = 0;
    double norm = tolerance + 1.0;

    while (iter < max_iter && norm > tolerance) {
        #pragma omp parallel for
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                new_grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
            }
        }

        norm = compute_norm(grid, new_grid, n);

        // Обновление сетки
        for (int i = 1; i < n-1; i++) {
            for (int j = 1; j < n-1; j++) {
                grid[i][j] = new_grid[i][j];
            }
        }

        iter++;
    }

    for (int i = 0; i < n; i++) {
        free(new_grid[i]);
    }
    free(new_grid);
}

int main() {
    double **grid = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        grid[i] = (double *)malloc(N * sizeof(double));
    }

    initialize_grid(grid, N);
    set_boundary_conditions(grid, N);

    gauss_seidel(grid, N, MAX_ITER, TOLERANCE);

    // Вывод результатов
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", grid[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < N; i++) {
        free(grid[i]);
    }
    free(grid);

    return 0;
}