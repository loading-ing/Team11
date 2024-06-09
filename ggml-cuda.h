#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// Always success. To check if CUDA is actually loaded, use `ggml_cublas_loaded`.
GGML_API void   ggml_init_cublas(void);//初始化cuBLAS库。该函数总是成功，可以通过ggml_cublas_loaded()检查CUDA是否成功加载。

// Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
GGML_API bool   ggml_cublas_loaded(void);//检查是否有可用的CUDA设备并且cuBLAS库是否成功加载。返回true表示成功，否则返回false。

GGML_API void * ggml_cuda_host_malloc(size_t size);//在CUDA主机上分配内存。返回指向分配的内存的指针。
GGML_API void   ggml_cuda_host_free(void * ptr);//释放由ggml_cuda_host_malloc()分配的CUDA主机内存。

//检查两个张量是否可以进行矩阵乘法，并将结果存储在dst张量中。返回true表示可以进行乘法，否则返回false。
GGML_API bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
GGML_API void   ggml_cuda_set_tensor_split(const float * tensor_split);//设置张量分割参数。
GGML_API void   ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor);//转换张量的数据。
GGML_API void   ggml_cuda_alloc_tensor(struct ggml_tensor * tensor);//为张量分配CUDA设备内存。
GGML_API void   ggml_cuda_free_data(struct ggml_tensor * tensor);//释放张量的数据。
GGML_API void   ggml_cuda_cpy_1d(struct ggml_tensor * dst, const struct ggml_tensor * src);//将一个张量的数据复制到另一个张量中。
GGML_API bool   debug_equal(short *a, short *b);//用于调试的相等性检查函数。
GGML_API void **ggml_cuda_get_data_pp(struct ggml_tensor * tensor);//获取张量的数据指针的指针。

GGML_API void   ggml_cuda_assign_buffers(struct ggml_tensor * tensor);//为张量分配缓冲区。
GGML_API void   ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor);//为张量分配缓冲区，不使用临时缓冲区。
GGML_API void   ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor * tensor);//为张量分配缓冲区，强制在原地进行操作。

GGML_API void   ggml_cuda_assign_buffers_no_alloc(struct ggml_tensor * tensor);//函数为给定的张量分配CUDA设备内存，但不进行实际的内存分配操作。
GGML_API void   ggml_cuda_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset);//函数为张量在CUDA设备内存中的偏移量设置指定的值。
GGML_API void   ggml_cuda_copy_to_device(struct ggml_tensor * tensor);//函数将张量数据从主机内存复制到CUDA设备内存。
GGML_API void   ggml_cuda_copy_to_host(struct ggml_tensor * tensor);//函数将张量数据从CUDA设备内存复制到主机内存。

GGML_API void   ggml_cuda_set_main_device(int main_device);//设置主设备。
GGML_API void   ggml_cuda_set_mul_mat_q(bool mul_mat_q);        //设置是否使用mul_mat_q函数。
GGML_API void   ggml_cuda_set_scratch_size(size_t scratch_size);        //设置临时缓冲区的大小。
GGML_API void   ggml_cuda_free_scratch(void);                  //释放临时缓冲区。
GGML_API bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);//函数执行张量的前向计算。

GGML_API int    ggml_cuda_get_device_count(void);       //获取可用CUDA设备的数量。
GGML_API void   ggml_cuda_get_device_description(int device, char * description, size_t description_size);//获取指定设备的描述信息。
GGML_API size_t ggml_cuda_get_free_memory(int device);      //获取指定设备的可用内存量。

GGML_API void   ggml_cuda_set_device_constants(float sparse_pred_threshold);        //函数为设备设置稀疏预测阈值。

// backend API
GGML_API ggml_backend_t ggml_backend_cuda_init(void); // TODO: take a list of devices to use

#ifdef  __cplusplus
}
#endif
