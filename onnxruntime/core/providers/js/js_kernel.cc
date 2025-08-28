// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "js_kernel.h"

EM_ASYNC_JS(
    int,                                                    // возврат в C++
    jsepKernelRun,                                          // имя функции
    (intptr_t kernel_handle, intptr_t serialized_ctx_ptr),  // параметры
    {
      // Module.jsepRunKernelAsync должна вернуть Promise<int>
      const status = await Module.jsepRunKernelAsync(
          Number(kernel_handle),
          Number(serialized_ctx_ptr),
          Module.jsepSessionState.sessionHandle,
          Module.jsepSessionState.errors);
      // Приведение к C‑инту.
      return Number(status);
    });

namespace onnxruntime {
namespace js {
}  // namespace js
}  // namespace onnxruntime
