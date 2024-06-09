#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_backend;
struct ggml_backend_buffer;

//
// Legacy API
//

typedef struct ggml_allocr * ggml_allocr_t;

// initialize allocator for use with CPU backend only
GGML_API ggml_allocr_t ggml_allocr_new(void * data, size_t size, size_t alignment);//使用给定的数据、大小和对齐方式初始化一个内存分配器
GGML_API ggml_allocr_t ggml_allocr_new_measure(size_t alignment);//使用给定的对齐方式初始化一个仅用于测量的内存分配器。

// initialize allocator for use with ggml-backend
GGML_API ggml_allocr_t ggml_allocr_new_from_buffer(struct ggml_backend_buffer * buffer);//使用给定的缓冲区初始化一个内存分配器，用于ggml后端。
GGML_API ggml_allocr_t ggml_allocr_new_from_backend(struct ggml_backend * backend, size_t size); // allocates an owned buffer
GGML_API ggml_allocr_t ggml_allocr_new_measure_from_backend(struct ggml_backend * backend);//使用给定的后端初始化一个仅用于测量的内存分配器。

GGML_API struct ggml_backend_buffer * ggml_allocr_get_buffer(ggml_allocr_t alloc);//获取内存分配器关联的缓冲区。

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
GGML_API void   ggml_allocr_set_parse_seq(ggml_allocr_t alloc, const int * list, int n);//告诉分配器按照给定的顺序解析节点。

GGML_API void   ggml_allocr_free       (ggml_allocr_t alloc);//释放内存分配器。
GGML_API bool   ggml_allocr_is_measure (ggml_allocr_t alloc);//检查内存分配器是否为测量分配器。
GGML_API void   ggml_allocr_reset      (ggml_allocr_t alloc);//重置内存分配器。
GGML_API void   ggml_allocr_alloc      (ggml_allocr_t alloc, struct ggml_tensor * tensor);//为给定的张量分配内存。
GGML_API size_t ggml_allocr_max_size   (ggml_allocr_t alloc);//获取内存分配器能够分配的最大大小。

GGML_API size_t ggml_allocr_alloc_graph(ggml_allocr_t alloc, struct ggml_cgraph * graph);//为给定的计算图分配内存。

//
// ggml-backend v2 API
//

// Seperate tensor and graph allocator objects
// This is necessary for multi-backend allocation because the graph allocator needs to use multiple tensor allocators
// The original API is kept as a wrapper around the new API

// Tensor allocator
typedef struct ggml_tallocr * ggml_tallocr_t;

GGML_API ggml_tallocr_t ggml_tallocr_new(void * data, size_t size, size_t alignment);//使用给定的数据、大小和对齐方式初始化一个张量内存分配器，
GGML_API ggml_tallocr_t ggml_tallocr_new_measure(size_t alignment);//使用给定的对齐方式初始化一个仅用于测量的张量内存分配器。
GGML_API ggml_tallocr_t ggml_tallocr_new_from_buffer(struct ggml_backend_buffer * buffer);//使用给定的缓冲区初始化一个张量内存分配器，用于ggml后端。
GGML_API ggml_tallocr_t ggml_tallocr_new_from_backend(struct ggml_backend * backend, size_t size); // allocates an owned buffer
GGML_API ggml_tallocr_t ggml_tallocr_new_measure_from_backend(struct ggml_backend * backend);//使用给定的后端初始化一个仅用于测量的张量内存分配器。

GGML_API struct ggml_backend_buffer * ggml_tallocr_get_buffer(ggml_tallocr_t talloc);//获取张量内存分配器关联的缓冲区。

GGML_API void   ggml_tallocr_free       (ggml_tallocr_t talloc);//释放张量内存分配器。
GGML_API bool   ggml_tallocr_is_measure (ggml_tallocr_t talloc);//检查张量内存分配器是否为测量分配器。
GGML_API void   ggml_tallocr_reset      (ggml_tallocr_t talloc);//重置张量内存分配器。
GGML_API void   ggml_tallocr_alloc      (ggml_tallocr_t talloc, struct ggml_tensor * tensor);//为给定的张量分配内存。
GGML_API size_t ggml_tallocr_max_size   (ggml_tallocr_t talloc);//获取张量内存分配器能够分配的最大大小。


// Graph allocator
typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(void);//初始化一个图形内存分配器。
GGML_API void   ggml_gallocr_free(ggml_gallocr_t galloc);//释放图形内存分配器。

GGML_API void   ggml_gallocr_set_parse_seq(ggml_gallocr_t galloc, const int * list, int n);//告诉图形内存分配器按照给定的顺序解析节点。
GGML_API size_t ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ggml_tallocr_t talloc, struct ggml_cgraph * graph);//为给定的计算图分配内存。

// Allocate tensors from the allocators given by the hash table
GGML_API void   ggml_gallocr_alloc_graph_n(
                    ggml_gallocr_t galloc,
                    struct ggml_cgraph * graph,
                    struct ggml_hash_set hash_set,
                    ggml_tallocr_t * hash_node_talloc);

#ifdef  __cplusplus
}
#endif
