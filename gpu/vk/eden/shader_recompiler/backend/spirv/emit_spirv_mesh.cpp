// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Switch 2 (NVN) task and mesh programs. They talk to the hardware through the stage buffer
// (ISBE): a task program writes the number of mesh workgroups at byte 4 of its output region and
// its payload after it; each mesh workgroup reads that region (its own index at byte 0) and
// writes its vertices and primitives to an attribute region in groups of 32, one 32-word line per
// attribute, plus the primitive count and 8-bit vertex indices to a map region. Here the task's
// output region is the task payload, the mesh output regions live in workgroup memory, and the
// epilogue (reached by every invocation, see structured_control_flow.cpp) hands them to Vulkan.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <span>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void Barrier(EmitContext& ctx) {
    ctx.OpControlBarrier(ctx.Const(static_cast<u32>(spv::Scope::Workgroup)),
                         ctx.Const(static_cast<u32>(spv::Scope::Workgroup)),
                         ctx.Const(static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease |
                                                    spv::MemorySemanticsMask::WorkgroupMemory |
                                                    (ctx.mesh_emu ? spv::MemorySemanticsMask::UniformMemory
                                                                  : spv::MemorySemanticsMask::MaskNone))));
}

Id LocalInvocation(EmitContext& ctx) {
    return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id), 0U);
}

Id Clamped(EmitContext& ctx, Id index, u32 count) {
    return ctx.OpUMin(ctx.U32[1], index, ctx.Const(count - 1));
}

Id PayloadWord(EmitContext& ctx, Id index) {
    const Id clamped{Clamped(ctx, index, ctx.mesh_info.payload_words)};
    if (ctx.mesh_emu) {
        // (the payload is global: the task and the mesh stage are two dispatches)
        return ctx.OpAccessChain(ctx.mesh_payload_u32, ctx.mesh_storage, ctx.Const(4U), clamped);
    }
    return ctx.OpAccessChain(ctx.mesh_payload_u32, ctx.mesh_payload, clamped);
}

Id SharedWord(EmitContext& ctx, Id array, Id index) {
    return ctx.OpAccessChain(ctx.mesh_shared_u32, array, index);
}

// Runs body under `if (condition)`.
template <typename Func>
void If(EmitContext& ctx, Id condition, Func&& body) {
    const Id then_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, then_label, merge_label);
    ctx.AddLabel(then_label);
    body();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

// Copies the lines of element `index` (a vertex or a primitive) from the attribute region.
void CopyLines(EmitContext& ctx, Id index, u32 base, u32 num_lines,
               const std::vector<EmitContext::MeshLineTarget>& targets, Id out_index = {}) {
    // Word of line 0: base + (index / 32) * num_lines * 32 + index % 32.
    const Id group{ctx.OpShiftRightLogical(ctx.U32[1], index, ctx.Const(5U))};
    const Id slot{ctx.OpBitwiseAnd(ctx.U32[1], index, ctx.Const(31U))};
    const Id first{ctx.OpIAdd(ctx.U32[1],
                              ctx.OpIMul(ctx.U32[1], group, ctx.Const(num_lines * 32)),
                              ctx.OpIAdd(ctx.U32[1], slot, ctx.Const(base)))};
    for (u32 line = 0; line < num_lines; ++line) {
        const EmitContext::MeshLineTarget& target{targets[line]};
        if (!Sirit::ValidId(target.variable)) {
            continue;
        }
        const Id word{ctx.OpLoad(
            ctx.U32[1],
            SharedWord(ctx, ctx.mesh_attributes, ctx.OpIAdd(ctx.U32[1], first, ctx.Const(line * 32))))};
        const Id value{target.is_float ? ctx.OpBitcast(ctx.F32[1], word) : word};
        const Id pointer_type{target.is_float ? (ctx.mesh_emu ? ctx.mesh_out_f32 : ctx.output_f32)
                                              : (ctx.mesh_emu ? ctx.mesh_out_u32 : ctx.output_u32)};
        Id array_base{target.variable};
        if (ctx.mesh_emu && Sirit::ValidId(target.member_pointer)) {
            array_base = ctx.OpAccessChain(target.member_pointer, target.variable, ctx.Const(target.member));
        }
        const Id output_index{Sirit::ValidId(out_index) ? out_index : index};
        const Id pointer{target.element == ~0U
                             ? ctx.OpAccessChain(pointer_type, array_base, output_index)
                             : ctx.OpAccessChain(pointer_type, array_base, output_index,
                                                 ctx.Const(target.element))};
        ctx.OpStore(pointer, value);
    }
}

// OpenGL-style clip space: z = (z + w) / 2, from the stage buffer's position lines.
void ConvertDepth(EmitContext& ctx, Id vertex, Id out_vertex = {}) {
    const auto& lines{ctx.mesh_info.vertex_lines};
    const auto z_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionZ))};
    const auto w_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionW))};
    if (z_line == lines.end() || w_line == lines.end()) {
        return;
    }
    const u32 num_lines{static_cast<u32>(lines.size())};
    const Id group{ctx.OpShiftRightLogical(ctx.U32[1], vertex, ctx.Const(5U))};
    const Id slot{ctx.OpBitwiseAnd(ctx.U32[1], vertex, ctx.Const(31U))};
    const Id first{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], group, ctx.Const(num_lines * 32)),
                              slot)};
    const auto read = [&](auto line) {
        const u32 offset{static_cast<u32>(line - lines.begin()) * 32};
        return ctx.OpBitcast(ctx.F32[1],
                             ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_attributes,
                                                               ctx.OpIAdd(ctx.U32[1], first,
                                                                          ctx.Const(offset)))));
    };
    const Id z{ctx.OpFMul(ctx.F32[1], ctx.OpFAdd(ctx.F32[1], read(z_line), read(w_line)),
                          ctx.Const(0.5f))};
    const Id target = ctx.mesh_emu
        ? ctx.OpAccessChain(ctx.mesh_out_f32, ctx.mesh_storage, ctx.Const(2U), out_vertex, ctx.Const(2U))
        : ctx.OpAccessChain(ctx.output_f32, ctx.output_position, vertex, ctx.Const(2U));
    ctx.OpStore(target, z);
}

u32 task_cap(u32 threads) {   // (RX_VK_TASK_CAP=<n>: mesh workgroups a task may start; 0: one a thread)
    static const char* e = getenv("RX_VK_TASK_CAP");
    const u32 n = e ? static_cast<u32>(atoi(e)) : 0U;
    return n ? n : threads;
}

void TaskEpilogue(EmitContext& ctx) {
    Barrier(ctx);
    // (hookshot) at most one mesh workgroup a task thread: each thread keeps or culls one meshlet (the programs compact them by subgroup
    // ballot). It was 65535, and a payload word a task left unwritten is anything: when the task culled everything, tens of thousands of
    // mesh workgroups for a draw held the GPU past Windows' two seconds, and the device was lost, on about half the first starts
    const Id count{ctx.OpUMin(ctx.U32[1], ctx.OpLoad(ctx.U32[1], PayloadWord(ctx, ctx.Const(1U))),
                              ctx.Const(task_cap(ctx.mesh_threads)))};
    const Id one{ctx.Const(1U)};
    if (ctx.mesh_emu) {
        // (the mesh dispatch reads {count, 1, 1} from the block's indirect member; one task workgroup a draw)
        const auto store = [&](u32 word, Id value) {
            const Id pointer{ctx.OpAccessChain(ctx.mesh_out_u32, ctx.mesh_storage, ctx.Const(1U),
                                               ctx.Const(word))};
            ctx.OpStore(pointer, value);
        };
        If(ctx, ctx.OpIEqual(ctx.U1, LocalInvocation(ctx), ctx.Const(0U)), [&] {
            store(0U, count);
            store(1U, one);
            store(2U, one);
        });
        return;
    }
    // Marker for OpEmitMeshTasksEXT count, 1, 1, payload (PatchMeshInstructions).
    ctx.OpStore(count, one);
    ctx.OpStore(one, ctx.mesh_payload);
    ctx.mesh_tasks_marker = {count, one, one, ctx.mesh_payload};
}

template <typename F>
void LoopOver(EmitContext& ctx, Id limit, F&& body) {
    // (a runtime loop: the unrolled form was 126 iterations, and the Adreno compiler rejected that module)
    ctx.OpStore(ctx.mesh_loop_counter, ctx.Const(0U));
    const Id header{ctx.OpLabel()};
    const Id check{ctx.OpLabel()};
    const Id body_label{ctx.OpLabel()};
    const Id cont{ctx.OpLabel()};
    const Id merge{ctx.OpLabel()};
    ctx.OpBranch(header);
    ctx.AddLabel(header);
    ctx.OpLoopMerge(merge, cont, spv::LoopControlMask::MaskNone);
    ctx.OpBranch(check);
    ctx.AddLabel(check);
    const Id value{ctx.OpLoad(ctx.U32[1], ctx.mesh_loop_counter)};
    ctx.OpBranchConditional(ctx.OpULessThan(ctx.U1, value, limit), body_label, merge);
    ctx.AddLabel(body_label);
    body(value);
    ctx.OpBranch(cont);
    ctx.AddLabel(cont);
    ctx.OpStore(ctx.mesh_loop_counter, ctx.OpIAdd(ctx.U32[1], value, ctx.Const(1U)));
    ctx.OpBranch(header);
    ctx.AddLabel(merge);
}

void MeshEpilogue(EmitContext& ctx) {
    const MeshInfo& mesh{ctx.mesh_info};
    Barrier(ctx);
    const Id count{ctx.OpUMin(ctx.U32[1], ctx.OpLoad(ctx.U32[1], ctx.mesh_primitive_count),
                              ctx.Const(mesh.max_primitives))};
    const Id max_vertices{ctx.Const(mesh.max_vertices)};
    if (!ctx.mesh_emu) {
        // Marker for OpSetMeshOutputsEXT max_vertices, count (PatchMeshInstructions).
        ctx.OpStore(max_vertices, count);
        ctx.mesh_outputs_marker = {max_vertices, count};
    }

    const Id invocation{LocalInvocation(ctx)};
    const u32 threads{ctx.mesh_threads};
    const bool convert_depth{ctx.runtime_info.convert_depth_mode &&
                             !ctx.profile.support_native_ndc &&
                             (ctx.mesh_emu || Sirit::ValidId(ctx.output_position))};
    // (mesh emulation: one vertex range and one index range a workgroup, taken from the block's counters once)
    Id vertex_base{}, index_base{};
    if (ctx.mesh_emu) {
        If(ctx, ctx.OpIEqual(ctx.U1, invocation, ctx.Const(0U)), [&] {
            const Id c0 = ctx.OpAccessChain(ctx.mesh_out_u32, ctx.mesh_storage, ctx.Const(0U),
                                            ctx.Const(0U));
            const Id c1 = ctx.OpAccessChain(ctx.mesh_out_u32, ctx.mesh_storage, ctx.Const(0U),
                                            ctx.Const(1U));
            const Id scope = ctx.Const(static_cast<u32>(spv::Scope::Device));
            const Id sem = ctx.Const(static_cast<u32>(
                spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory));
            const Id v = ctx.OpAtomicIAdd(ctx.U32[1], c0, scope, sem, max_vertices);
            const Id i = ctx.OpAtomicIAdd(ctx.U32[1], c1, scope, sem, count);
            ctx.OpStore(ctx.mesh_vertex_base, v);
            ctx.OpStore(ctx.mesh_index_base, i);
        });
        Barrier(ctx);
        vertex_base = ctx.OpLoad(ctx.U32[1], ctx.mesh_vertex_base);
        index_base = ctx.OpLoad(ctx.U32[1], ctx.mesh_index_base);
    }
    const u32 vertex_lines{static_cast<u32>(mesh.vertex_lines.size())};
    const u32 primitive_lines{static_cast<u32>(mesh.primitive_lines.size())};
    const auto copy_vertex = [&](Id vertex) {
        const Id out_index{ctx.mesh_emu ? ctx.OpIAdd(ctx.U32[1], vertex_base, vertex) : Id{}};
        CopyLines(ctx, vertex, 0, vertex_lines, ctx.mesh_vertex_targets, out_index);
        if (convert_depth) {
            ConvertDepth(ctx, vertex, out_index);
        }
    };
    const auto copy_primitive = [&](Id primitive) {
        // Vertex indices: map bytes 4 + 3 * primitive (lines: 4 + 2 * primitive, points: 4 + primitive).
        const u32 corners{mesh.Corners()};
        const Id byte{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], primitive, ctx.Const(corners)),
                                 ctx.Const(4U))};
        std::array<Id, 3> indices{ctx.Const(0U), ctx.Const(0U), ctx.Const(0U)};
        for (u32 element = 0; element < corners; ++element) {
            const Id address{ctx.OpIAdd(ctx.U32[1], byte, ctx.Const(element))};
            const Id raw = ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_map, address));
            indices[element] = ctx.mesh_emu ? raw : Clamped(ctx, raw, mesh.max_vertices);
        }
        if (ctx.mesh_emu) {
            // (the indices point into the compacted vertex list, not into this workgroup's stage buffer)
            for (u32 element = 0; element < corners; ++element) {
                indices[element] = ctx.OpSelect(ctx.U32[1],
                    ctx.OpULessThan(ctx.U1, indices[element], max_vertices),
                    ctx.OpIAdd(ctx.U32[1], vertex_base, indices[element]), ctx.Const(~0U));
            }
        }
        Id indices_base{ctx.mesh_primitive_indices};
        if (ctx.mesh_emu && Sirit::ValidId(ctx.mesh_primitive_pointer)) {
            indices_base = ctx.OpAccessChain(ctx.mesh_primitive_pointer, ctx.mesh_primitive_indices,
                                             ctx.Const(ctx.mesh_primitive_member));
        }
        const Id index_slot{ctx.mesh_emu ? ctx.OpIAdd(ctx.U32[1], index_base, primitive) : primitive};
        const Id output_indices{ctx.TypePointer(
            ctx.mesh_emu ? spv::StorageClass::StorageBuffer : spv::StorageClass::Output,
            ctx.U32[ctx.mesh_emu ? 3 : corners])};
        ctx.OpStore(ctx.OpAccessChain(output_indices, indices_base, index_slot),
                    ctx.mesh_emu ? ctx.OpCompositeConstruct(ctx.U32[3], std::span<const Id>(indices.data(), 3))
                    : corners == 1 ? indices[0]   // (a point's index is a scalar)
                                 : ctx.OpCompositeConstruct(ctx.U32[corners],
                                                            std::span<const Id>(indices.data(), corners)));
        CopyLines(ctx, primitive, ctx.mesh_primitive_base, primitive_lines,
                  ctx.mesh_primitive_targets);
    };
    if (threads == 1) {
        LoopOver(ctx, max_vertices, copy_vertex);
        LoopOver(ctx, count, copy_primitive);
    } else {
        for (u32 first = 0; first < mesh.max_vertices; first += threads) {
            const Id vertex{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
            if (first + threads > mesh.max_vertices) {
                If(ctx, ctx.OpULessThan(ctx.U1, vertex, max_vertices), [&] { copy_vertex(vertex); });
            } else {
                copy_vertex(vertex);
            }
        }
        for (u32 first = 0; first < mesh.max_primitives; first += threads) {
            const Id primitive{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
            If(ctx, ctx.OpULessThan(ctx.U1, primitive, count), [&] { copy_primitive(primitive); });
        }
    }
}

} // Anonymous namespace

Id EmitIsbeRead(EmitContext& ctx, Id address) {
    const Id workgroup{
        ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), 0U)};
    if (ctx.stage == Stage::Task) {
        if (ctx.mesh_emu) {
            // One task workgroup per region. Host writes its absolute first task at byte 8.
            const Id first = ctx.OpLoad(ctx.U32[1], ctx.OpAccessChain(
                ctx.mesh_out_u32, ctx.mesh_storage, ctx.Const(0U), ctx.Const(2U)));
            return ctx.OpIAdd(ctx.U32[1], workgroup, first);
        }
        if (ctx.wants_mesh_first && Sirit::ValidId(ctx.rescaling_push_constants)) {
            // (hookshot) + the draw's first workgroup: NVN draws [first, first + count). Without it every draw began at 0: the
            // foliage's draws (first up to ~1900) drew the first records again and again, and the forest had no leaves
            const Id table{ctx.OpLoad(ctx.U64, ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U64),
                                                                  ctx.rescaling_push_constants, ctx.Const(ctx.mesh_first_member_index)))};
            const Id draw{ctx.OpLoad(ctx.U32[1], ctx.draw_index)};
            const Id at{ctx.OpIAdd(ctx.U64, table, ctx.OpUConvert(ctx.U64, ctx.OpIMul(ctx.U32[1], draw, ctx.Const(16U))))};
            const Id member{ctx.OpAccessChain(ctx.mesh_first_member_pointer, ctx.OpConvertUToPtr(ctx.mesh_first_block_pointer, at),
                                              ctx.Const(0U))};
            const Id first{ctx.OpLoad(ctx.U32[1], member, spv::MemoryAccessMask::Aligned, 4U)};
            return ctx.OpIAdd(ctx.U32[1], workgroup, first);
        }
        return workgroup; // the input region holds the workgroup index (VERTEX_ID)
    }
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], address, ctx.Const(2U))};
    const Id word{ctx.OpLoad(ctx.U32[1], PayloadWord(ctx, index))};
    return ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, address, ctx.u32_zero_value), workgroup,
                        word);
}

void EmitIsbeWriteAttribute(EmitContext& ctx, Id address, Id value) {
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], address, ctx.Const(2U))};
    if (ctx.stage == Stage::Task) {
        ctx.OpStore(PayloadWord(ctx, index), value);
        return;
    }
    ctx.OpStore(SharedWord(ctx, ctx.mesh_attributes, Clamped(ctx, index, ctx.mesh_attribute_words)),
                value);
}

void EmitIsbeWriteMapU8(EmitContext& ctx, Id address, Id value) {
    if (ctx.stage != Stage::Mesh) {
        throw NotImplementedException("Stage buffer map write outside mesh programs");
    }
    ctx.OpStore(SharedWord(ctx, ctx.mesh_map, Clamped(ctx, address, ctx.mesh_map_bytes)),
                ctx.OpBitwiseAnd(ctx.U32[1], value, ctx.Const(0xffU)));
}

void EmitIsbeWriteMapU32(EmitContext& ctx, const IR::Value& address, Id value) {
    // The primitive count: the map is written from its end, so its first word is at byte 3.
    if (ctx.stage != Stage::Mesh || !address.IsImmediate() || address.U32() != 3) {
        throw NotImplementedException("32-bit stage buffer map write");
    }
    ctx.OpStore(ctx.mesh_primitive_count, value);
}

void EmitMeshPrologue(EmitContext& ctx) {
    if (ctx.stage == Stage::Task) {
        // (hookshot) the count starts at nothing: task payload memory holds whatever the last task left, and a task that keeps no meshlet
        // never writes it. The barrier: every thread has done this before any of them goes on to write the real count
        if (ctx.mesh_emu) {
            If(ctx, ctx.OpIEqual(ctx.U1, LocalInvocation(ctx), ctx.Const(0U)), [&] {
                ctx.OpStore(PayloadWord(ctx, ctx.Const(1U)), ctx.u32_zero_value);
            });
        } else {
            ctx.OpStore(PayloadWord(ctx, ctx.Const(1U)), ctx.u32_zero_value);
        }
        Barrier(ctx);
    }
    if (ctx.stage != Stage::Mesh) {
        return;
    }
    // (local) The stage buffer's map and attribute regions start at zero, as they come out on NVIDIA: workgroup memory holds
    // whatever the last workgroup on that unit left, and a vertex or an index byte the program left unwritten was that. On an
    // RX 7700S the foliage drawn into the 2400 x 2400 shadow map around Link came out with hundreds of long thin triangles
    // across the whole map (vertices from other meshlets' leftovers), dark lines on the ground, the water and Link that
    // changed every frame. Zeroed, such a primitive is (0,0,0): no area. Straight-line stores, each clamped into its array
    // (no branch: see below). RX_VK_MESH_NO_ZERO=1: as before
    static const bool no_zero{std::getenv("RX_VK_MESH_NO_ZERO") != nullptr};
    if (!no_zero && ctx.mesh_threads) {
        const Id invocation{LocalInvocation(ctx)};
        const auto zero_all = [&](Id array, u32 count) {
            if (!Sirit::ValidId(array) || count == 0) {
                return;
            }
            for (u32 first = 0; first < count; first += ctx.mesh_threads) {
                const Id at{Clamped(ctx, ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first)), count)};
                ctx.OpStore(SharedWord(ctx, array, at), ctx.u32_zero_value);
            }
        };
        zero_all(ctx.mesh_map, ctx.mesh_map_bytes);
        zero_all(ctx.mesh_attributes, ctx.mesh_attribute_words);
    }
    // Every invocation stores the zero: no branch here, the entry block's label is a phi
    // predecessor for later blocks.
    ctx.OpStore(ctx.mesh_primitive_count, ctx.u32_zero_value);
    Barrier(ctx);
}

void EmitMeshEpilogue(EmitContext& ctx) {
    if (ctx.stage == Stage::Task) {
        TaskEpilogue(ctx);
    } else if (ctx.stage == Stage::Mesh) {
        if (ctx.mesh_emu) {
            // Defensive write bound. Runtime must split dispatches to <=64 children;
            // flag an invalid dispatch instead of corrupting the next region.
            const Id group = ctx.OpLoad(ctx.U32[3], ctx.workgroup_id);
            Id fits = ctx.OpULessThan(ctx.U1, ctx.OpCompositeExtract(ctx.U32[1], group, 0U), ctx.Const(64U));
            for (u32 axis = 1; axis < 3; ++axis)
                fits = ctx.OpLogicalAnd(ctx.U1, fits, ctx.OpIEqual(ctx.U1,
                    ctx.OpCompositeExtract(ctx.U32[1], group, axis), ctx.Const(0U)));
            If(ctx, fits, [&] { MeshEpilogue(ctx); });
            If(ctx, ctx.OpLogicalAnd(ctx.U1, ctx.OpLogicalNot(ctx.U1, fits),
                                     ctx.OpIEqual(ctx.U1, LocalInvocation(ctx), ctx.Const(0U))), [&] {
                const Id flag = ctx.OpAccessChain(ctx.mesh_out_u32, ctx.mesh_storage, ctx.Const(0U), ctx.Const(3U));
                ctx.OpAtomicOr(ctx.U32[1], flag, ctx.Const(static_cast<u32>(spv::Scope::Device)),
                               ctx.Const(static_cast<u32>(spv::MemorySemanticsMask::MaskNone)), ctx.Const(1U));
            });
        } else {
            MeshEpilogue(ctx);
        }
    }
}

} // namespace Shader::Backend::SPIRV
