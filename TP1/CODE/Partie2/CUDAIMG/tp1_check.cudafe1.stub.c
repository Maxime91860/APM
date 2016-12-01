#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "tp1_check.fatbin.c"
extern void __device_stub__Z6kernelPdS_S_i(double *, double *, double *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_17_tp1_check_cpp1_ii_8a4ba4bc(void) __attribute__((__constructor__));
void __device_stub__Z6kernelPdS_S_i(double *__par0, double *__par1, double *__par2, int __par3){__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaLaunch(((char *)((void ( *)(double *, double *, double *, int))kernel)));}
# 5 "tp1_check.cu"
void kernel( double *__cuda_0,double *__cuda_1,double *__cuda_2,int __cuda_3)
# 6 "tp1_check.cu"
{__device_stub__Z6kernelPdS_S_i( __cuda_0,__cuda_1,__cuda_2,__cuda_3);
# 13 "tp1_check.cu"
}
# 1 "CUDAIMG/tp1_check.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T21) {  __nv_dummy_param_ref(__T21); __nv_save_fatbinhandle_for_managed_rt(__T21); __cudaRegisterEntry(__T21, ((void ( *)(double *, double *, double *, int))kernel), _Z6kernelPdS_S_i, (-1)); }
static void __sti____cudaRegisterAll_17_tp1_check_cpp1_ii_8a4ba4bc(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
