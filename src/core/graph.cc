#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        auto rebuildConnections = [this]() {
            // 清空图中已有的反向连接
            for (const auto &tensor : tensors) {
                tensor->targets.clear();
                tensor->source.reset();
            }
            for (const auto &op : ops) {
                op->predecessors.clear();
                op->successors.clear();
            }
            // 建立 Tensor->source
            for (const auto &op : ops) {
                for (const auto &output : op->getOutputs())
                    if (output)
                        output->source = op;  // 一个 Tensor 只能由一个 Operator 产生。（但一个 Tensor 可以被多个 Operator 使用。）
            }
            // 建立 Tensor->targets, Operator.successors, Operator.predecessors
            for (const auto &op : ops) {
                for (const auto &input : op->getInputs()) {
                    if (!input) continue;
                    input->targets.emplace_back(op);
                    if (auto producer = input->source.lock()) {
                        producer->successors.emplace_back(op);
                        op->predecessors.emplace_back(producer);
                    }
                }
            }
            sorted = false;
        };

        auto eraseOperator = [this](const Operator &op) {
            ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
        };

        auto eraseTensor = [this](const Tensor &tensor) {
            tensors.erase(std::remove(tensors.begin(), tensors.end(), tensor), tensors.end());
        };

        // 判断两个连续的 Transpose 是否互为逆操作
        auto areInversePermutations = [](const vector<int> &first, const vector<int> &second) {
            if (first.size() != second.size())
                return false;
            for (size_t i = 0; i < first.size(); i++) {
                if (second[i] < 0 || second[i] >= static_cast<int>(first.size()))
                    return false;
                if (first[second[i]] != static_cast<int>(i))
                    return false;
            }
            return true;
        };
        
        auto isLastTwoAxesSwap = [](const Ref<TransposeObj> &transpose) {
            const vector<int> permute = transpose->getPermute();
            const size_t rank = permute.size();

            if (rank < 2) return false;

            for (size_t i = 0; i < rank; i++) {
                int expected = static_cast<int>(i);
                if (i == rank - 2)
                    expected = static_cast<int>(rank - 1);
                else if (i == rank - 1)
                    expected = static_cast<int>(rank - 2);
                if (permute[i] != expected)
                    return false;
            }
            return true;
        };

        rebuildConnections();

        // 规则一
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto &outerOp : ops) {
                auto outer = as<TransposeObj>(outerOp);  // 第二个 Transpose
                if (!outer) continue;
                Tensor middle = outer->getInputs(0);
                if (!middle) continue;
                auto inner = as<TransposeObj>(middle->getSource());  // 第一个 Transpose
                if (!inner) continue;

                // inner 的输出只能被 outer 消费，否则不能安全删除 inner
                const auto consumers = middle->getTargets();
                if (consumers.size() != 1 || consumers[0] != outerOp)
                    continue;

                if (!areInversePermutations(inner->getPermute(), outer->getPermute()))
                    continue;
                
                Tensor original = inner->getInputs(0);
                Tensor replaced = outer->getOutput();
                for (const auto &op : ops)
                    if (op.get() != inner.get() && op.get() != outer.get())  // op 不是 inner 也不是 outer 的情况下才能替换
                        op->replaceInput(replaced, original);
                eraseOperator(inner);
                eraseOperator(outer);
                eraseTensor(middle);
                eraseTensor(replaced);
                
                rebuildConnections();
                changed = true;
                break;
            }
        }
        // 规则二
        changed = true;
        while (changed) {
            changed = false;
            for (const auto &matmulOp : ops) {
                auto matmul = as<MatmulObj>(matmulOp);
                if (!matmul) continue;
                for (size_t i = 0; i < matmul->getInputs().size(); i++) {
                    Tensor transposed = matmul->getInputs(i);  // 先取 matmul 的一个输入 tensor
                    if (!transposed) continue;
                    auto transpose = as<TransposeObj>(transposed->getSource());  // 尝试把产生这个输入tensor的operator当成TransposeObj。as<TransposeObj>(someOperator)大致等价于std::dynamic_pointer_cast<TransposeObj>(someOperator)，它会检查 someOperator 的真实动态类型，若是 TransposeObj，就返回有效的 Ref<TransposeObj>，否则返回 nullptr
                    if (!transpose || !isLastTwoAxesSwap(transpose))  // 若它确实是Transpose，才做融合
                        continue;
                    // 只有 Transpose 的结果没有被其他算子共享时才能安全融合
                    const auto consumers = transposed->getTargets();
                    if (consumers.size() != 1 || consumers[0] != matmulOp)
                        continue;
                    
                    Tensor original = transpose->getInputs(0);
                    matmul->replaceInput(transposed, original);
                    if (i == 0)
                        matmul->setTransA(!matmul->getTransA());
                    else
                        matmul->setTransB(!matmul->getTransB());
                    eraseOperator(transpose);
                    eraseTensor(transposed);
                    
                    rebuildConnections();
                    changed = true;
                    break;
                }
                if (changed) break;
            }
        }
        rebuildConnections();
        IT_ASSERT(topo_sort());
        shape_infer();
        IT_ASSERT(checkValid());
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        std::unordered_map<Tensor, size_t> offsets;
        std::unordered_map<Tensor, size_t> remainingUses;
        for (const auto &tensor : tensors)
            remainingUses.emplace(tensor, tensor->getTargets().size());  // 第二维是剩余使用次数
        auto allocateTensor = [&](const Tensor &tensor) {
            IT_ASSERT(tensor != nullptr);
            if (offsets.find(tensor) == offsets.end())
                offsets.emplace(tensor, allocator.alloc(tensor->getBytes()));
        };
        for (const auto &input : getInputs())
            allocateTensor(input);
        for (const auto &op : ops) {
            for (const auto &input : op->getInputs())
                allocateTensor(input);
            for (const auto &output : op->getOutputs())
                allocateTensor(output);
            for (const auto &input : op->getInputs()) { // 当前算子已经完成对某输入tensor的最后一次使用时，逻辑上释放该输入
                auto useIt = remainingUses.find(input);
                IT_ASSERT(useIt != remainingUses.end());
                IT_ASSERT(useIt->second > 0);
                --useIt->second;
                if (useIt->second == 0)
                    allocator.free(offsets.at(input), input->getBytes());
            }
        }
        // 在逻辑内存规划完成、获得所有 offset 后，真正申请一次连续内存
        auto *base = static_cast<char *>(allocator.getPtr());
        for (const auto &[tensor, offset] : offsets)
            tensor->setDataBlob(make_ref<BlobObj>(runtime, static_cast<void *>(base + offset)));  // BlobObj 可以理解为“某段设备内存的句柄”
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini