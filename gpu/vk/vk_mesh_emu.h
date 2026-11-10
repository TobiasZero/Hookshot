// Mesh-shader emulation, the runtime half (Phase 3, docs/arm64-phase3.md).
//
// A device without VK_EXT_mesh_shader runs the translated task/mesh program as a compute dispatch that writes one
// region of this ring (vertices, indices, the indirect command). The draw then binds the ring and issues one indexed
// indirect draw through the pass-through vertex shader (mesh_passthrough_vs.cpp).
//
// The ring is created once, when the device is created, and lives for the run. Regions are handed out per draw and
// never freed: the ring wraps, and a wrap must not pass a region an earlier submission still reads (see the layout
// header for the size arithmetic).
#pragma once

#include "vk_internal.h"
#include "vk_mesh_emu_layout.h"

namespace mkwvk {

struct MeshEmuRegion {
    uint64_t offset = 0;      // (start of the region in the ring)
    uint64_t index_at = 0;    // (the index buffer binding's offset)
    uint32_t vertices = 0;
    uint32_t primitives = 0;
};

/// Creates the ring when the device takes the emulation path; false (and no ring) otherwise.
bool mesh_emu_init();
bool mesh_emu_ready();
void mesh_emu_shutdown();

/// The next region, or offset kMeshEmuNoRoom when the ring cannot hold it.
MeshEmuRegion mesh_emu_alloc(uint32_t vertices, uint32_t primitives);

/// Set 1 of an emulated mesh pipeline: binding 0 is the ring, a dynamic storage buffer. Created once; VK_NULL_HANDLE
/// when the ring does not exist (a device with mesh shaders).
VkDescriptorSetLayout mesh_emu_set_layout();
VkDescriptorSet mesh_emu_set();

/// Binds set 1 of the pipeline layout with the region's byte offset.
void mesh_emu_bind(VkCommandBuffer cmd, VkPipelineLayout layout, VkPipelineBindPoint bind_point,
                   uint64_t region_offset);

/// Clears `size` bytes at `offset` in the region (the counters and the indirect command, before a dispatch).
void mesh_emu_clear(VkCommandBuffer cmd, const MeshEmuRegion& region, uint32_t offset, uint32_t size,
                    uint32_t first_task = 0);

/// Binds the compute pipeline and its ring set, then dispatches `groups` workgroups.
void mesh_emu_dispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                       const MeshEmuRegion& region, uint32_t groups);

/// The task stage: dispatches `x, y, z` workgroups; it writes the mesh workgroup count into the region.
void mesh_emu_task_dispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                            const MeshEmuRegion& region, uint32_t x, uint32_t y, uint32_t z);

/// The mesh stage after a task: waits for it, then dispatches the count the task wrote (indirect).
void mesh_emu_dispatch_indirect(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                                const MeshEmuRegion& region);

/// One invocation of the finalize pass: the index counter becomes the indirect draw command.
void mesh_emu_finalize(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                       const MeshEmuRegion& region);

/// Compute wrote the region; the graphics draw reads it. Records the draw. The caller has dispatched already, recorded
/// mesh_emu_compute_done() outside the render pass, and binds the pass-through pipeline's descriptor set.
void mesh_emu_draw_indirect(VkCommandBuffer cmd, const MeshEmuRegion& region, VkPipeline pipeline);

/// Outside a render pass, before the draw's clears and dispatches: earlier draws that read ring memory (a wrapped
/// region) finish before it is written again (write-after-read).
void mesh_emu_compute_begin(VkCommandBuffer cmd);

/// Outside a render pass, after the draw's dispatches: the compute writes become visible to the indirect draw, the index
/// fetch and the pass-through vertex shader. (A barrier with a compute source is not allowed inside a render pass.)
void mesh_emu_compute_done(VkCommandBuffer cmd);

} // namespace mkwvk
