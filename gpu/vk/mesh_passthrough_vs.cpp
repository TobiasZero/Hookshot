// SPDX-License-Identifier: GPL-2.0-or-later
// Generated with Sirit: see mesh_passthrough_vs.h.

#include "mesh_passthrough_vs.h"
#include <stdexcept>

#include <sirit/sirit.h>

namespace mkwvk {
namespace {

using Sirit::Id;

} // Anonymous namespace

std::vector<uint32_t> mesh_passthrough_vs() {
    Sirit::Module module{0x00010300};   // (SPIR-V 1.3: what every Vulkan 1.1+ device accepts)
    module.AddCapability(spv::Capability::Shader);
    module.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);

    const Id void_t{module.TypeVoid()};
    const Id fn_t{module.TypeFunction(void_t)};
    const Id u32{module.TypeInt(32, false)};
    const Id f32{module.TypeFloat(32)};
    const Id vec4{module.TypeVector(f32, 4)};
    const Id i32{module.TypeInt(32, true)};
    const Id zero{module.Constant(i32, 0)};

    // ---- the region: a 64-byte header (counters and the indirect command), then the positions ----
    const Id header{module.TypeArray(u32, module.Constant(u32, 16U))};
    module.Decorate(header, spv::Decoration::ArrayStride, 4U);
    const Id positions{module.TypeRuntimeArray(vec4)};
    module.Decorate(positions, spv::Decoration::ArrayStride, 16U);
    const Id block{module.TypeStruct(header, positions)};
    module.Decorate(block, spv::Decoration::Block);
    module.MemberDecorate(block, 0U, spv::Decoration::Offset, 0U);
    module.MemberDecorate(block, 1U, spv::Decoration::Offset, 64U);

    const Id ssbo_ptr{module.TypePointer(spv::StorageClass::StorageBuffer, block)};
    const Id ssbo{module.AddGlobalVariable(ssbo_ptr, spv::StorageClass::StorageBuffer)};
    module.Decorate(ssbo, spv::Decoration::DescriptorSet, 1U);
    module.Decorate(ssbo, spv::Decoration::Binding, 0U);

    // ---- builtins ----
    const Id vertex_index_ptr{module.TypePointer(spv::StorageClass::Input, i32)};
    const Id vertex_index{module.AddGlobalVariable(vertex_index_ptr, spv::StorageClass::Input)};
    module.Decorate(vertex_index, spv::Decoration::BuiltIn, spv::BuiltIn::VertexIndex);

    const Id position_ptr{module.TypePointer(spv::StorageClass::Output, vec4)};
    const Id position{module.AddGlobalVariable(position_ptr, spv::StorageClass::Output)};
    module.Decorate(position, spv::Decoration::BuiltIn, spv::BuiltIn::Position);

    // ---- main: gl_Position = records[gl_VertexIndex].position ----
    const Id main_fn{module.OpFunction(void_t, spv::FunctionControlMask::MaskNone, fn_t)};
    module.AddLabel();
    const Id index{module.OpLoad(i32, vertex_index)};
    const Id one{module.Constant(i32, 1)};
    const Id element{module.OpAccessChain(module.TypePointer(spv::StorageClass::StorageBuffer, vec4), ssbo, one, index)};
    const Id value{module.OpLoad(vec4, element)};
    module.OpStore(position, value);
    module.OpReturn();
    module.OpFunctionEnd();

    module.AddEntryPoint(spv::ExecutionModel::Vertex, main_fn, "main", vertex_index, position);
    return module.Assemble();
}

std::vector<uint32_t> mesh_finalize_spv(uint32_t counter_word, uint32_t indirect_word,
                                         uint32_t max_vertices, uint32_t max_primitives,
                                         uint32_t corners) {
    if (!max_vertices || !max_primitives || corners < 1 || corners > 3)
        throw std::invalid_argument("invalid mesh emulation capacity/topology");
    Sirit::Module m{0x00010300};
    m.AddCapability(spv::Capability::Shader);
    m.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    const Id void_t=m.TypeVoid(), fn=m.TypeFunction(void_t), u=m.TypeInt(32,false), b=m.TypeBool();
    auto c=[&](uint32_t x){return m.Constant(u,x);};
    const Id words=m.TypeRuntimeArray(u);
    m.Decorate(words,spv::Decoration::ArrayStride,4U);
    const Id block=m.TypeStruct(words);
    m.Decorate(block,spv::Decoration::Block);
    m.MemberDecorate(block,0U,spv::Decoration::Offset,0U);
    const Id ssbo=m.AddGlobalVariable(m.TypePointer(spv::StorageClass::StorageBuffer,block),spv::StorageClass::StorageBuffer);
    m.Decorate(ssbo,spv::Decoration::DescriptorSet,1U);
    m.Decorate(ssbo,spv::Decoration::Binding,0U);
    const Id ptr=m.TypePointer(spv::StorageClass::StorageBuffer,u);
    auto at=[&](Id x){return m.OpAccessChain(ptr,ssbo,c(0),x);};
    const Id main=m.OpFunction(void_t,spv::FunctionControlMask::MaskNone,fn);
    m.AddLabel();
    const Id local=m.TypePointer(spv::StorageClass::Function,u);
    const Id primitive=m.AddLocalVariable(local,spv::StorageClass::Function);
    const Id output=m.AddLocalVariable(local,spv::StorageClass::Function);
    m.OpStore(primitive,c(0)); m.OpStore(output,c(0));
    const Id vc=m.OpUMin(u,m.OpLoad(u,at(c(0))),c(max_vertices));
    const Id pc=m.OpUMin(u,m.OpLoad(u,at(c(counter_word))),c(max_primitives));
    const Id loop=m.OpLabel(), body=m.OpLabel(), next=m.OpLabel(), done=m.OpLabel();
    m.OpBranch(loop); m.AddLabel(loop);
    const Id p=m.OpLoad(u,primitive);
    const Id more=m.OpULessThan(b,p,pc);
    m.OpLoopMerge(done,next,spv::LoopControlMask::MaskNone);
    m.OpBranchConditional(more,body,done); m.AddLabel(body);
    // Source primitive slots are 16 bytes, including for point and line output.
    // Read ALL corners before writing: compacted output overlaps earlier slots.
    const Id source=m.OpIAdd(u,c(16+max_vertices*4),m.OpIMul(u,p,c(4)));
    Id indices[3]{}; Id valid=m.ConstantTrue(b);
    for(uint32_t k=0;k<corners;k++) {
        indices[k]=m.OpLoad(u,at(m.OpIAdd(u,source,c(k))));
        valid=m.OpLogicalAnd(b,valid,m.OpULessThan(b,indices[k],vc));
    }
    const Id keep=m.OpLabel(), merge=m.OpLabel();
    m.OpSelectionMerge(merge,spv::SelectionControlMask::MaskNone);
    m.OpBranchConditional(valid,keep,merge); m.AddLabel(keep);
    const Id n=m.OpLoad(u,output);
    for(uint32_t k=0;k<corners;k++)
        m.OpStore(at(m.OpIAdd(u,c(16+max_vertices*4+k),n)),indices[k]);
    m.OpStore(output,m.OpIAdd(u,n,c(corners)));
    m.OpBranch(merge); m.AddLabel(merge); m.OpBranch(next); m.AddLabel(next);
    m.OpStore(primitive,m.OpIAdd(u,p,c(1))); m.OpBranch(loop); m.AddLabel(done);
    const Id values[5]{m.OpLoad(u,output),c(1),c(0),c(0),c(0)};
    for(uint32_t k=0;k<5;k++) m.OpStore(at(c(indirect_word+k)),values[k]);
    m.OpReturn(); m.OpFunctionEnd();
    m.AddEntryPoint(spv::ExecutionModel::GLCompute,main,"main");
    m.AddExecutionMode(main,spv::ExecutionMode::LocalSize,1U,1U,1U);
    return m.Assemble();
}

} // namespace mkwvk
