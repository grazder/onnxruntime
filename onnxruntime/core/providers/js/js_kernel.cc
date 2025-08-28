// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "js_kernel.h"

namespace onnxruntime {
namespace js {
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
}  // namespace js
}  // namespace onnxruntime
