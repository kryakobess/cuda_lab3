#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024

#define G 6.67E-11

const int nthreads = 4;
const char* fileName = "input.csv";

void calculate_force(float *masses, float *array_x, float *array_y, float* fx, float *fy, int n) 
{
    for(int i = 0; i < n; ++i) {
        fx[i] = 0.0;
        fy[i] = 0.0;
    }
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


void parse_csv(const char *filename, int n, float *m, float *x, float *y, float *vx, float *vy) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;

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
            count++;
        }
    }

    fclose(file);
}


int main(int argc, char* argv[]) 
{
    int n; // кол-во тел
    float t_end = 100.0; // максимальный промежуток времени
    float delta_t = t_end / 100.0;

    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return EXIT_FAILURE;
    }

    // Read the first line to get the value of n
    if (fscanf(file, "%d", &n) != 1) {
        fprintf(stderr, "Failed to read the value of n\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    fclose(file);

    // for(int i = 0; i < n; ++i) 
    // {
    //     scanf("%f %f %f %f %f", &masses[i], &array_x[i], &array_y[i], &vs_x[i], &vs_y[i]);
    // }
    float *masses = malloc(n * sizeof(float));
    float *array_x = malloc(n * sizeof(float));
    float *array_y = malloc(n * sizeof(float));
    float *vs_x = malloc(n * sizeof(float));
    float *vs_y = malloc(n * sizeof(float));
    float *fx = malloc(n * sizeof(float));
    float *fy = malloc(n * sizeof(float));
    parse_csv(fileName, n, masses, array_x, array_y, vs_x, vs_y);

    float current_time = 0.0;
    FILE *res;
    res = fopen("result.csv", "w+");

    float res_arr[n][3];

    while(current_time < t_end) {
        fprintf(res, "%f ", current_time);
        for(int i = 0; i < n; ++i) {
            res_arr[i][0] = current_time;
            res_arr[i][1] = array_x[i];
            res_arr[i][]
            fprintf(res, "%f %f ", array_x[i], array_y[i]);
            //printf("%f %f ", array_x[i], array_y[i]);
        }
        //printf("\n");
        fprintf(res, "\n");
        calculate_force(masses, array_x, array_y, fx, fy, n);
        update_points(fx,  fy,  masses,  array_x,  array_y, vs_x, vs_y, n, delta_t);
        current_time += delta_t;
    }
    fclose(res);
    printf("Result saved in file 'result.csv'!\n");

    free(masses);
    free(array_x);
    free(array_y);
    free(fx);
    free(fy);
    return 0;
}