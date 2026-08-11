#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        IT_ASSERT(inputs.size() == 2);

        const Shape &shapeA = inputs[0]->getDims();
        const Shape &shapeB = inputs[1]->getDims();

        IT_ASSERT(shapeA.size() >= 2);
        IT_ASSERT(shapeB.size() >= 2);

        const size_t rankA = shapeA.size();
        const size_t rankB = shapeB.size();

        m = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
        const int kA = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];

        const int kB = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        n = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];

        IT_ASSERT(kA == kB, "Matmul dimensions K do not match");
        k = kA;

        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);
        Shape outputShape = infer_broadcast(batchA, batchB);

        outputShape.emplace_back(m);
        outputShape.emplace_back(n);

        return {{outputShape}};
    }

} // namespace infini