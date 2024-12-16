#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define G 6.67E-11

const int nthreads = 4;

__device__ void update_points(float *fx, float* fy, float *masses, float *array_x, float *array_y,
 float *v_x, float *v_y, int n, float delta_t) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    array_x[i] += v_x[i] * delta_t;
    array_y[i] += v_y[i] * delta_t;
    v_x[i] += (fx[i] / masses[i]) * delta_t;
    v_y[i] += (fy[i] / masses[i]) * delta_t;
}

__global__ void calculate_force(float *fx, float* fy, float *masses, float *array_x, float *array_y,
 float *v_x, float *v_y, int n, float delta_t) 
{
    int my_idx = blockDim.x * blockIdx.x + threadIdx.x;
    fx[my_idx] = 0.0;
    fy[my_idx] = 0.0;

    for (int i = 0; i < n; ++i) {
        if (i == my_idx) continue;

        float dx = array_x[i] - array_x[my_idx];
        float dy = array_y[i] - array_y[my_idx];
        float squared_dist = dx*dx + dy*dy;
        float dist = sqrtf(squared_dist);
        float force = G * masses[my_idx] * masses[i] / (squared_dist * dist);
        fx[my_idx] += force * dx;
        fy[my_idx] += force * dy;
    }

    update_points(fx, fy, masses, array_x, array_y, v_x, v_y, n, delta_t);
}

__host__ void freeMem(float *masses, float *array_x, float *array_y, float *vs_x, float *vs_y, float *fx, float *fy) {
    cudaFree(masses);
    cudaFree(array_x);
    cudaFree(array_y);
    cudaFree(vs_x);
    cudaFree(vs_y);
    cudaFree(fx);
    cudaFree(fy);
}

void generate_bodies(float *masses, float *array_x, float *array_y, float *v_x, float *v_y, int n) {
    for(int i = 0; i < n; ++i) {
        masses[i] = ((float) rand()) / (RAND_MAX >> 10); 
        array_x[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
        array_y[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
        v_x[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
        v_y[i] = 2.0 * ((float) rand()) / RAND_MAX - 1.0;
        printf("Generating body: i=%d m=%f x=%f y=%f vx=%f vy=%f\n", i, masses[i], array_x[i], array_y[i], v_x[i], v_y[i]);
    }
}

__host__ void parse_csv_and_init(const char *filename, int n, float *masses, float *array_x,
 float *array_y, float *vs_x, float *vs_y, float* fx, float* fy) {
    printf("Parsing csv %s\n", filename);

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file");
        exit(EXIT_FAILURE);
    }
    printf("File is opened\n");

    if (fscanf(file, "%d", &n) != 1) {
        printf("Failed to read the number of rows\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    printf("n=%d\n", n);

    cudaMallocManaged(&masses, n*sizeof(float));
    cudaMallocManaged(&array_x, n * sizeof(float));
    cudaMallocManaged(&array_y, n * sizeof(float));
    cudaMallocManaged(&vs_x, n * sizeof(float));
    cudaMallocManaged(&vs_y, n * sizeof(float));
    cudaMallocManaged(&fx, n * sizeof(float));
    cudaMallocManaged(&fy, n * sizeof(float));
    
    // Пропуск строки заголовка
    char header[1024];
    fgets(header, sizeof(header), file);
    if (fgets(header, sizeof(header), file) == NULL) {
        printf("Failed to read header\n");
        fclose(file);
        freeMem(masses, array_x, array_y, vs_x, vs_y, fx, fy);
        exit(EXIT_FAILURE);
    }
    printf("%s\n", header);

    // Чтение данных
    char buf[1024];
    for (int i = 0; i < n; ++i) {
        fgets(buf, sizeof(buf), file);
        printf("%s \n", buf);
        if (sscanf(buf, "%f;%f;%f;%f;%f;", &masses[i], &array_x[i], &array_y[i], &vs_x[i], &vs_y[i]) != 5) {
            printf("Failed to read data at row %d\n", i + 1);
            fclose(file);
            freeMem(masses, array_x, array_y, vs_x, vs_y, fx, fy);
            exit(EXIT_FAILURE);
        }
        printf("i=%d m=%f x=%f y=%f vx=%f vy=%f\n", i, masses[i], array_x[i], array_y[i], vs_x[i], vs_y[i]);
    }

    fclose(file);
}

int main(int argc, char* argv[]) 
{
    printf("Start\n");
    
    int n; // кол-во тел и потоков
    float t_end = 10.0; // максимальный промежуток времени
    float time_step_count = 100.0;
    float delta_t = t_end / time_step_count;
    int block_cnt = 1;

    float *masses;
    float *array_x;
    float *array_y;
    float *vs_x;
    float *vs_y;
    float *fx;
    float *fy;

    parse_csv_and_init("input.csv", n, masses, array_x, array_y, vs_x, vs_y, fx, fy);
    //generate_bodies(masses, array_x, array_y, vs_x, vs_y, n);

    float current_time = 0.0;
    while(current_time < t_end) {
        printf("%f ", current_time);
        for(int i = 0; i < n; ++i) {
            printf("%f %f ", array_x[i], array_y[i]);
        }
        printf("\n");
        calculate_force<<<block_cnt, n>>>(fx,  fy,  masses,  array_x,  array_y, vs_x, vs_y, n, delta_t);
        cudaDeviceSynchronize();
        current_time += delta_t;
    }

    cudaDeviceSynchronize();

    freeMem( masses, array_x, array_y, vs_x, vs_y, fx, fy);

    return 0;
}