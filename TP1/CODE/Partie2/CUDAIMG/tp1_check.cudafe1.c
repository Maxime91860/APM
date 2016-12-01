# 1 "tp1_check.cu"
# 56 "/usr/local/cuda-7.5/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic push


#pragma GCC diagnostic ignored "-Wunused-function"
# 35 "/usr/include/c++/4.8/exception" 3
#pragma GCC visibility push ( default )
# 149 "/usr/include/c++/4.8/exception" 3
#pragma GCC visibility pop
# 42 "/usr/include/c++/4.8/new" 3
#pragma GCC visibility push ( default )
# 120 "/usr/include/c++/4.8/new" 3
#pragma GCC visibility pop
# 1888 "/usr/local/cuda-7.5/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 1425 "/usr/local/cuda-7.5/bin/../targets/x86_64-linux/include/driver_types.h"
struct CUstream_st;
# 180 "/usr/include/libio.h" 3
enum __codecvt_result {

__codecvt_ok,
__codecvt_partial,
__codecvt_error,
__codecvt_noconv};
# 51 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
enum idtype_t {
P_ALL,
P_PID,
P_PGID};
# 190 "/usr/include/math.h" 3
enum _ZUt_ {
FP_NAN,


FP_INFINITE,


FP_ZERO,


FP_SUBNORMAL,


FP_NORMAL};
# 302 "/usr/include/math.h" 3
enum _LIB_VERSION_TYPE {
_IEEE_ = (-1),
_SVID_,
_XOPEN_,
_POSIX_,
_ISOC_};
# 128 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_voidIvEUt_E { _ZNSt9__is_voidIvE7__valueE = 1};
# 148 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIbEUt_E { _ZNSt12__is_integerIbE7__valueE = 1};
# 155 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIcEUt_E { _ZNSt12__is_integerIcE7__valueE = 1};
# 162 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIaEUt_E { _ZNSt12__is_integerIaE7__valueE = 1};
# 169 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIhEUt_E { _ZNSt12__is_integerIhE7__valueE = 1};
# 177 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIwEUt_E { _ZNSt12__is_integerIwE7__valueE = 1};
# 201 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIsEUt_E { _ZNSt12__is_integerIsE7__valueE = 1};
# 208 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerItEUt_E { _ZNSt12__is_integerItE7__valueE = 1};
# 215 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIiEUt_E { _ZNSt12__is_integerIiE7__valueE = 1};
# 222 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIjEUt_E { _ZNSt12__is_integerIjE7__valueE = 1};
# 229 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIlEUt_E { _ZNSt12__is_integerIlE7__valueE = 1};
# 236 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerImEUt_E { _ZNSt12__is_integerImE7__valueE = 1};
# 243 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIxEUt_E { _ZNSt12__is_integerIxE7__valueE = 1};
# 250 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIyEUt_E { _ZNSt12__is_integerIyE7__valueE = 1};
# 268 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIfEUt_E { _ZNSt13__is_floatingIfE7__valueE = 1};
# 275 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIdEUt_E { _ZNSt13__is_floatingIdE7__valueE = 1};
# 282 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIeEUt_E { _ZNSt13__is_floatingIeE7__valueE = 1};
# 358 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIcEUt_E { _ZNSt9__is_charIcE7__valueE = 1};
# 366 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIwEUt_E { _ZNSt9__is_charIwE7__valueE = 1};
# 381 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIcEUt_E { _ZNSt9__is_byteIcE7__valueE = 1};
# 388 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIaEUt_E { _ZNSt9__is_byteIaE7__valueE = 1};
# 395 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIhEUt_E { _ZNSt9__is_byteIhE7__valueE = 1};
# 138 "/usr/include/c++/4.8/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIeEUt_E { _ZNSt12__is_integerIeE7__valueE}; enum _ZNSt12__is_integerIdEUt_E { _ZNSt12__is_integerIdE7__valueE}; enum _ZNSt12__is_integerIfEUt_E { _ZNSt12__is_integerIfE7__valueE};
# 212 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3
typedef unsigned long size_t;
#include "crt/host_runtime.h"
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 2782 "/usr/local/cuda-7.5/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern enum cudaError cudaConfigureCall(struct dim3, struct dim3, size_t, struct CUstream_st *);
# 3999 "/usr/local/cuda-7.5/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern enum cudaError cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind);
# 15 "tp1_check.cu"
extern int main(int, char **);
extern int __cudaSetupArgSimple();
extern int __cudaLaunch();
extern void __nv_dummy_param_ref();
extern void __nv_save_fatbinhandle_for_managed_rt();
extern int __cudaRegisterEntry();
extern int __cudaRegisterBinary();
static void __sti___17_tp1_check_cpp1_ii_8a4ba4bc(void) __attribute__((__constructor__));
# 15 "tp1_check.cu"
int main( int argc,  char **argv)
{  unsigned __T20;
 int __cuda_local_var_42667_9_non_const_N;
 int __cuda_local_var_42668_9_non_const_sz_in_bytes;

 double *__cuda_local_var_42670_13_non_const_h_a;
# 20 "tp1_check.cu"
 double *__cuda_local_var_42670_19_non_const_h_b;
# 20 "tp1_check.cu"
 double *__cuda_local_var_42670_25_non_const_h_c;
 double *__cuda_local_var_42671_13_non_const_d_a;
# 21 "tp1_check.cu"
 double *__cuda_local_var_42671_19_non_const_d_b;
# 21 "tp1_check.cu"
 double *__cuda_local_var_42671_25_non_const_d_c;
# 43 "tp1_check.cu"
 struct dim3 __cuda_local_var_42693_10_non_const_dimBlock;
 struct dim3 __cuda_local_var_42694_10_non_const_dimGrid;
# 55 "tp1_check.cu"
 double __cuda_local_var_42705_12_non_const_err;
# 55 "tp1_check.cu"
 double __cuda_local_var_42705_21_non_const_norm;
# 17 "tp1_check.cu"
__cuda_local_var_42667_9_non_const_N = 1000;
__cuda_local_var_42668_9_non_const_sz_in_bytes = ((int)(((unsigned long)__cuda_local_var_42667_9_non_const_N) * 8UL));




__cuda_local_var_42670_13_non_const_h_a = ((double *)(malloc(((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes))));
__cuda_local_var_42670_19_non_const_h_b = ((double *)(malloc(((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes))));
__cuda_local_var_42670_25_non_const_h_c = ((double *)(malloc(((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes)))); {


 int i;
# 28 "tp1_check.cu"
i = 0; for (; (i < __cuda_local_var_42667_9_non_const_N); i++)
{
(__cuda_local_var_42670_13_non_const_h_a[i]) = ((1.0) / ((1.0) + ((double)i)));
(__cuda_local_var_42670_19_non_const_h_b[i]) = ((((double)i) - (1.0)) / (((double)i) + (1.0)));
} }


cudaMalloc(((void **)(&__cuda_local_var_42671_13_non_const_d_a)), ((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes));
cudaMalloc(((void **)(&__cuda_local_var_42671_19_non_const_d_b)), ((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes));
cudaMalloc(((void **)(&__cuda_local_var_42671_25_non_const_d_c)), ((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes));


cudaMemcpy(((void *)__cuda_local_var_42671_13_non_const_d_a), ((const void *)__cuda_local_var_42670_13_non_const_h_a), ((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes), cudaMemcpyHostToDevice);
cudaMemcpy(((void *)__cuda_local_var_42671_19_non_const_d_b), ((const void *)__cuda_local_var_42670_19_non_const_h_b), ((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes), cudaMemcpyHostToDevice);

{
# 421 "/usr/local/cuda-7.5/bin/../targets/x86_64-linux/include/vector_types.h"
(__cuda_local_var_42693_10_non_const_dimBlock.x) = 64U; (__cuda_local_var_42693_10_non_const_dimBlock.y) = 1U; (__cuda_local_var_42693_10_non_const_dimBlock.z) = 1U;
# 43 "tp1_check.cu"
}
{ __T20 = (((((unsigned)__cuda_local_var_42667_9_non_const_N) + (__cuda_local_var_42693_10_non_const_dimBlock.x)) - 1U) / (__cuda_local_var_42693_10_non_const_dimBlock.x));
# 421 "/usr/local/cuda-7.5/bin/../targets/x86_64-linux/include/vector_types.h"
{ (__cuda_local_var_42694_10_non_const_dimGrid.x) = __T20; (__cuda_local_var_42694_10_non_const_dimGrid.y) = 1U; (__cuda_local_var_42694_10_non_const_dimGrid.z) = 1U; }
# 44 "tp1_check.cu"
}
(cudaConfigureCall(__cuda_local_var_42694_10_non_const_dimGrid, __cuda_local_var_42693_10_non_const_dimBlock, 0UL, ((struct CUstream_st *)0LL))) ? ((void)0) : (__device_stub__Z6kernelPdS_S_i(__cuda_local_var_42671_13_non_const_d_a, __cuda_local_var_42671_19_non_const_d_b, __cuda_local_var_42671_25_non_const_d_c, __cuda_local_var_42667_9_non_const_N));

cudaMemcpy(((void *)__cuda_local_var_42670_25_non_const_h_c), ((const void *)__cuda_local_var_42671_25_non_const_d_c), ((size_t)__cuda_local_var_42668_9_non_const_sz_in_bytes), cudaMemcpyDeviceToHost);


cudaFree(((void *)__cuda_local_var_42671_13_non_const_d_a));
cudaFree(((void *)__cuda_local_var_42671_19_non_const_d_b));
cudaFree(((void *)__cuda_local_var_42671_25_non_const_d_c));


__cuda_local_var_42705_12_non_const_err = (0.0); __cuda_local_var_42705_21_non_const_norm = (0.0); {
 int i;
# 56 "tp1_check.cu"
i = 0; for (; (i < __cuda_local_var_42667_9_non_const_N); i++)
{
 double __cuda_local_var_42708_16_non_const_err_loc;
# 58 "tp1_check.cu"
__cuda_local_var_42708_16_non_const_err_loc = (fabs(((__cuda_local_var_42670_25_non_const_h_c[i]) - ((__cuda_local_var_42670_13_non_const_h_a[i]) + (__cuda_local_var_42670_19_non_const_h_b[i])))));
__cuda_local_var_42705_12_non_const_err += __cuda_local_var_42708_16_non_const_err_loc;
__cuda_local_var_42705_21_non_const_norm += (fabs((__cuda_local_var_42670_25_non_const_h_c[i])));
} }
printf(((const char *)"Relative error : %.3e\n"), (__cuda_local_var_42705_12_non_const_err / __cuda_local_var_42705_21_non_const_norm));

free(((void *)__cuda_local_var_42670_13_non_const_h_a));
free(((void *)__cuda_local_var_42670_19_non_const_h_b));
free(((void *)__cuda_local_var_42670_25_non_const_h_c));

return 0;
}
static void __sti___17_tp1_check_cpp1_ii_8a4ba4bc(void) {   }

#include "CUDAIMG/tp1_check.cudafe1.stub.c"
