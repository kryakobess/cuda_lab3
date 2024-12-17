#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define G 6.67E-11

const int nthreads = 4;

__device__ void update_points(float *fx, float* fy, float *masses, float *array_x, float *array_y,
 float *v_x, float *v_y, int n, float delta_t) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("update points idx: %d\n", i);

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

    //printf("Calculate force. idx: %d\n", my_idx);
    for (int i = 0; i < n; ++i) {
        if (i == my_idx) continue;

        float dx = array_x[i] - array_x[my_idx];
        float dy = array_y[i] - array_y[my_idx];

        //printf("idx: %d, i: %d, dx: %f, dy: %f, xi: %f, yi: %f\n", my_idx, i, dx, dy, array_x[i], array_y[i]);
        
        float squared_dist = dx*dx + dy*dy;
        float dist = sqrtf(squared_dist);
        float force = G * masses[my_idx] * masses[i] / (squared_dist * dist);
        
        //printf("idx: %d, i: %d, force: %f", my_idx, i, force);

        fx[my_idx] += force * dx;
        fy[my_idx] += force * dy;
        //printf("idx: %d, i: %d, fx=%f, fy=%f\n", my_idx, i, fx[my_idx], fy[my_idx]);
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

void parse_csv(const char *filename, int n, float *m, float *x, float *y, float *vx, float *vy) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;

    printf("Parsing csv:\n");

    // Skip the first lines n and header
    fgets(line, sizeof(line), file);
    fgets(line, sizeof(line), file);
    
    while (fgets(line, sizeof(line), file) && count < n) {
        char *token = strtok(line, ";");
        if (token != NULL) {
            m[count] = atof(token);
            token = strtok(NULL, ";");
            x[count] = atof(token);
            token = strtok(NULL, ";");
            y[count] = atof(token);
            token = strtok(NULL, ";");
            vx[count] = atof(token);
            token = strtok(NULL, ";");
            vy[count] = atof(token);
            printf("%d : %f %f %f %f %f\n", count, m[count], x[count], y[count], vx[count], vy[count]);
            count++;
        }
    }

    fclose(file);
}

int main(int argc, char* argv[]) 
{
    printf("Start\n");
    
    int n; // кол-во тел и потоков
    float t_end = 100.0; // максимальный промежуток времени
    float time_step_count = 100.0;
    float delta_t = t_end / time_step_count;
    int block_cnt = 1;

    FILE *file = fopen("input.csv", "r");
    if (file == NULL) {
        perror("Failed to open file");
        return EXIT_FAILURE;
    }

    if (fscanf(file, "%d", &n) != 1) {
        fprintf(stderr, "Failed to read the value of n\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    fclose(file);

    float *masses;
    float *array_x;
    float *array_y;
    float *vs_x;
    float *vs_y;
    float *fx;
    float *fy;

    cudaMallocManaged(&masses, n*sizeof(float));
    cudaMallocManaged(&array_x, n * sizeof(float));
    cudaMallocManaged(&array_y, n * sizeof(float));
    cudaMallocManaged(&vs_x, n * sizeof(float));
    cudaMallocManaged(&vs_y, n * sizeof(float));
    cudaMallocManaged(&fx, n * sizeof(float));
    cudaMallocManaged(&fy, n * sizeof(float));

    parse_csv("input.csv", n, masses, array_x, array_y, vs_x, vs_y);

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