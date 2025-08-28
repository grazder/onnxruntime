// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "js_kernel.h"

EM_ASYNC_JS(
    int, __jsepKernelRun_impl,
    (intptr_t kernel_handle, intptr_t serialized_ctx_ptr),
    {
      const status = await Module.jsepRunKernelAsync(
          Number(kernel_handle),
          Number(serialized_ctx_ptr),
          Module.jsepSessionState.sessionHandle,
          Module.jsepSessionState.errors);
      return Number(status);
    });

extern "C" {
static inline int jsepKernelRun(intptr_t kernel_handle,
                                intptr_t serialized_ctx_ptr) {
  return __jsepKernelRun_impl(kernel_handle, serialized_ctx_ptr);
}
}  // extern "C"

namespace onnxruntime {
namespace js {

}  // namespace js
}  // namespace onnxruntime
