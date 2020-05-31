/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/kernel2.cc
 * \brief New kernels
 */
#include "./kernel2.h"

#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace kernel {
namespace {

// Check whether the given arguments have the same context.
inline void CheckCtx(
    const DLContext& ctx,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (aten::IsNullArray(arrays[i]))
      continue;
    CHECK_EQ(ctx, arrays[i]->ctx)
      << "Expected device context " << ctx << ". But got "
      << arrays[i]->ctx << " for " << names[i] << ".";
  }
}

}  // namespace

void SpMM(const std::string& op, const std::string& reduce,
          const UnitGraph* graph,
          NDArray ufeat,
          NDArray efeat,
          NDArray out,
          std::vector<NDArray> out_aux,
          SparseFormat format) {
  // TODO(minjie): fmt tuning
  format = SparseFormat::kCSR;
  ATEN_XPU_SWITCH(graph->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH(ufeat->dtype, DType, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SpMMCsr<XPU, IdType, DType>(op, reduce, graph->GetCSRMatrix(0),
                                      ufeat, efeat, out, out_aux);
        } else if (format == SparseFormat::kCOO) {
          SpMMCoo<XPU, IdType, DType>(op, reduce, graph->GetCOOMatrix(0),
                                      ufeat, efeat, out, out_aux);
        } else {
          LOG(FATAL) << "SpMM only supports CSR and COO foramts";
        }
      });
    });
  });
}

void SDDMM(const std::string& op,
           const UnitGraph* graph,
           NDArray ufeat,
           NDArray efeat,
           NDArray out,
           std::vector<NDArray> out_aux,
           SparseFormat format) {
  // TODO(minjie): fmt tuning
  format = SparseFormat::kCOO;
  ATEN_XPU_SWITCH(graph->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH(ufeat->dtype, DType, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SDDMMCsr<XPU, IdType, DType>(op, graph->GetCSRMatrix(0),
                                       ufeat, efeat, out, out_aux);
        } else if (format == SparseFormat::kCOO) {
          SDDMMCoo<XPU, IdType, DType>(op, graph->GetCOOMatrix(0),
                                       ufeat, efeat, out, out_aux);
        } else {
          LOG(FATAL) << "SpMM only supports CSR and COO foramts";
        }
      });
    });
  });
}

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpESum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    CheckCtx(graph->Context(), {X, Y, Z}, {"U_data", "E_data", "Out"});
    CHECK_EQ(graph->NumEdgeTypes(), 1);
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyUSum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {X, Z}, {"U_data", "Out"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyUMax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    NDArray argX = args[3];
    CheckCtx(graph->Context(), {X, Z, argX}, {"U_data", "Out", "U_index"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyUMin")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    NDArray argX = args[3];
    CheckCtx(graph->Context(), {X, Z, argX}, {"U_data", "Out", "U_index"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyESum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray Y = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {Y, Z}, {"E_data", "Out"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyEMax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray Y = args[1];
    NDArray Z = args[2];
    NDArray argY = args[3];
    CheckCtx(graph->Context(), {Y, Z, argY}, {"E_data", "Out", "E_index"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyEMin")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray Y = args[1];
    NDArray Z = args[2];
    NDArray argY = args[3];
    CheckCtx(graph->Context(), {Y, Z, argY}, {"E_data", "Out", "E_index"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelCopyU")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray X = args[1];
    NDArray Z = args[2];
    CheckCtx(graph->Context(), {X, Z}, {"U_data", "Out"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpEMax")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    NDArray argX = args[5];
    NDArray argY = args[6];
    CheckCtx(graph->Context(), {X, Y, Z, argX, argY},
        {"U_data", "E_data", "Out", "U_index", "E_index"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpEMin")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    NDArray argX = args[5];
    NDArray argY = args[6];
    CheckCtx(graph->Context(), {X, Y, Z, argX, argY},
        {"U_data", "E_data", "Out", "U_index", "E_index"});
  });

DGL_REGISTER_GLOBAL("kernel2._CAPI_DGLKernelUOpV")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    HeteroGraphRef graph = args[1];
    NDArray X = args[2];
    NDArray Y = args[3];
    NDArray Z = args[4];
    CheckCtx(graph->Context(), {X, Y, Z}, {"U_data", "V_data", "Out"});
  });

}  // namespace kernel
}  // namespace dgl
