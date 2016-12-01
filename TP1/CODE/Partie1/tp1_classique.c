#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void kernel (double *a, double *b, double *c, int N, int blockDim_x, int gridDim_x)
{
    //Fonction C classique
    // int i;
    // for(i=0; i<N; i++){
    //     c[i] = a[i] + b[i];
    // }

    int i,j;

    for(i=0; i<gridDim_x; i++){
        for(j=0; j<blockDim_x; j++){
            int indice = j + blockDim_x * i; 
            if (indice < N){
                c[indice] = a[indice] + b[indice];
            }
        }
    }
}

void affiche(double* tab, int N, int width){
    int i;
    printf("(");
    for(i=0; i<N; i++){
        if(i != 0 && (i+1)%width == 0){
            printf("%f)\n",tab[i]);
            if(i != (N-1)){
                printf("(");
            }
        }
        else{
                printf("%f ",tab[i]);
        }
    }
}

int main(int argc, char **argv)
{
    int N = 1000;
    int sz_in_bytes = N*sizeof(double);

    double *h_a, *h_b, *h_c;
    int i;

    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);


    // Initiate values on h_a and h_b
    for(i = 0 ; i < N ; i++)
    {
    	h_a[i] = 1./(1.+i);
    	h_b[i] = (i-1.)/(i+1.);
    }

    kernel(h_a, h_b, h_c, N, 1024, 16);

    affiche(h_c, N, 10);


    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
