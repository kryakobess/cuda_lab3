#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define G 6.67E-11

const int nthreads = 4;


__global__ void calculate_force(float *masses, float *array_x, float *array_y, float* fx, float *fy, int n) 
{
    int my_idx = blockDim.x * blockIdx.x + threadIdx.x;
    fx[my_idx] = 0.0;
    fy[my_idx] = 0.0;

    float *local_fx[nthreads];
    float *local_fy[nthreads];
    for(int i = 0; i < nthreads; ++i) {
        local_fx[i] = calloc(n, sizeof(float));
        local_fy[i] = calloc(n, sizeof(float));
    }

#pragma omp parallel num_threads(nthreads) 
{
    int rank = omp_get_thread_num();
#pragma omp for
    for(int i = 0; i < n; ++i){
        for(int j = i+1; j < n; ++j) {
            float dx = array_x[j] - array_x[i];
            float dy = array_y[j] - array_y[i];
            float squared_dist = dx*dx + dy*dy;
            float dist = sqrtf(squared_dist);
            float force = G * masses[i] * masses[j] / (squared_dist * dist);
            local_fx[rank][i] += force * dx;
            local_fy[rank][i] += force * dy;
            local_fx[rank][j] -= force * dx;
            local_fy[rank][j] -= force * dy;
        }
    }
#pragma omp for
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < nthreads; ++j) {
            fx[i] += local_fx[j][i];
            fy[i] += local_fy[j][i];
        }
    }
}
    
    for(int i = 0; i < nthreads; ++i) {
        free(local_fx[i]);
        free(local_fy[i]);
    }
}

__device__ calculate_all_forces_to_body(float *masses, float *array_x, float *array_y, float* fx, float *fy, int n) {

}

void update_points(float *fx, float* fy, float *masses, float *array_x, float *array_y,
 float *v_x, float *v_y, int n, float delta_t) 
{
#pragma omp parallel for
    for(int i = 0; i < n; ++i) 
    {
        array_x[i] += v_x[i] * delta_t;
        array_y[i] += v_y[i] * delta_t;
        v_x[i] += (fx[i] / masses[i]) * delta_t;
        v_y[i] += (fy[i] / masses[i]) * delta_t;
    }
}

// void generate_bodies(float *masses, float *array_x, float *array_y, float *v_x, float *v_y, int n) {
//     for(int i = 0; i < n; ++i) {
//         masses[i] = ((float) rand()) / (RAND_MAX >> 10); 
//         array_x[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
//         array_y[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
//         v_x[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
//         v_y[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
//     }
// }

__host__ void parse_csv_and_init(const char *filename, int *n, float *masses, float *array_x, float *array_y, float *v_x, float *v_y) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    cudaMallocManaged(&n, 1);

    if (fscanf(file, "%d", n) != 1) {
        fprintf(stderr, "Failed to read the number of rows\n");
        fclose(file);
        freeMem(n, masses, array_x, array_y, vs_x, vs_y);
        exit(EXIT_FAILURE);
    }

    cudaMallocManaged(&masses, *n * sizeof(float));
    cudaMallocManaged(&array_x, *n * sizeof(float));
    cudaMallocManaged(&array_y, *n * sizeof(float));
    cudaMallocManaged(&vs_x, *n * sizeof(float));
    cudaMallocManaged(&vs_y, *n * sizeof(float));
    cudaMallocManaged(&fx, *n * sizeof(float));
    cudaMallocManaged(&fy, *n * sizeof(float));

    // Пропуск строки заголовка
    char header[1024];
    if (fgets(header, sizeof(header), file) == NULL) {
        fprintf(stderr, "Failed to read header\n");
        fclose(file);
        freeMem(n, masses, array_x, array_y, vs_x, vs_y);
        exit(EXIT_FAILURE);
    }

    // Чтение данных
    for (int i = 0; i < *n; ++i) {
        if (fscanf(file, "%f;%f;%f;%f;%f;", masses[i], array_x[i], array_y[i], v_x[i], v_y[i]) != 5) {
            fprintf(stderr, "Failed to read data at row %d\n", i + 1);
            fclose(file);
            freeMem(n, masses, array_x, array_y, vs_x, vs_y);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

__host__ void freeMem(int *n, float *masses, float *array_x, float *array_y, float *v_x, float *v_y) {
    cudaFree(n);
    cudaFree(masses);
    cudaFree(array_x);
    cudaFree(array_y);
    cudaFree(vs_x);
    cudaFree(vs_y);
    cudaFree(fx);
    cudaFree(fy);
}

int main(int argc, char* argv[]) 
{
    int *n; // кол-во тел и потоков
    float t_end = 10.0; // максимальный промежуток времени
    float time_step_count = 100.0;
    float *masses = malloc();
    float *array_x;
    float *array_y;
    float *vs_x;
    float *vs_y;
    float *fx;
    float *fy;
    float *times
    __managed__ float delta_t = t_end / time_step_count;

    parse_csv_and_init("input.csv", n, masses, array_x, array_y, vs_x, vs_y);

    float current_time = 0.0;
    //while(current_time < t_end) {
        printf("%f ", current_time);
        for(int i = 0; i < n; ++i) {
            printf("%f %f ", array_x[i], array_y[i]);
        }
        printf("\n");
        calculate_force(masses, array_x, array_y, fx, fy, n);
        update_points(fx,  fy,  masses,  array_x,  array_y, vs_x, vs_y, n, delta_t);
        current_time += delta_t;
    //}

    cudaDeviceSynchronize();

    freeMem(n, masses, array_x, array_y, vs_x, vs_y);

    return 0;
}