#include <stdlib.h>
#define ARRAY_SIZE 100*4032

extern void get_array_size(int* rows,int* cols);

// Returns the number of rows and columns of the data array in rows and cols variables
void get_array_size(int* rows,int* cols)
{
    (*rows) = 100;
    (*cols) = 4032;
}

extern double* read_data();



// Returns the data array could be two-dimensional also
double* read_data()
{
    double* data = (double*)malloc(sizeof(double)*ARRAY_SIZE); 
    for(int i=0;i<ARRAY_SIZE;i++)
    {
        data[i] = i;
    }

    return data;
}

// Helper function to free the allocated memory from Python when done with the data
void release_data(double* data)
{
    free(data);
}