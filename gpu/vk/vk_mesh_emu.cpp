// SPDX-License-Identifier: GPL-2.0-or-later
// Mesh-shader emulation, the runtime half: the ring and the indirect draw (vk_mesh_emu.h).

#include "vk_mesh_emu.h"
#include "vk_texture.h"

#include <cstring>
#include <mutex>

namespace mkwvk {
namespace {

std::mutex ring_mu;
VkBuffer ring = VK_NULL_HANDLE;
VmaAllocation ring_alloc = nullptr;
void* ring_map = nullptr;
uint64_t ring_cursor = 0;
bool ring_ready = false;

constexpr VkBufferUsageFlags kRingUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT;
constexpr uint64_t kMeshEmuPayloadBytes = 16384ull * sizeof(uint32_t);
constexpr uint64_t kMeshEmuMaxRegionBytes = 16ull << 20;

} // Anonymous namespace

bool mesh_emu_ready() {
    return ring_ready;
}

bool mesh_emu_init() {
    if (!g_vk_emulate_mesh_shader) {
        return false;
    }
    if (ring_ready) {
        return true;
    }
    // (host-visible and mapped: the CPU seeds the header, the compute shader writes the rest in place. On the UMA part
    // this is the same physical memory the GPU reads; no staging buffer)
    if (!create_buffer_cached(kMeshEmuRingBytes, kRingUsage, ring, ring_alloc, &ring_map, true) || !ring_map) {
        logf("vk: mesh emulation: the %llu MB ring could not be created", (unsigned long long)(kMeshEmuRingBytes >> 20));
        return false;
    }
    ring_cursor = 0;
    ring_ready = true;
    logf("vk: mesh emulation: a %llu MB ring of vertex, index and indirect records", (unsigned long long)(kMeshEmuRingBytes >> 20));
    return true;
}

void mesh_emu_shutdown() {
    std::lock_guard lk(ring_mu);
    if (!ring) {
        return;
    }
    destroy_buffer(ring, ring_alloc);
    ring = VK_NULL_HANDLE;
    ring_alloc = nullptr;
    ring_map = nullptr;
    ring_ready = false;
}

MeshEmuRegion mesh_emu_alloc(uint32_t vertices, uint32_t primitives) {
    MeshEmuRegion region;
    const uint64_t bytes = mesh_emu_region_bytes(vertices, primitives) + kMeshEmuPayloadBytes;
    const uint64_t usable_ring = kMeshEmuRingBytes - kMeshEmuMaxRegionBytes;
    if (bytes > kMeshEmuMaxRegionBytes || bytes > usable_ring) {
        region.offset = kMeshEmuNoRoom;   // (0 is a valid region: say "no room" explicitly)
        return region;
    }
    std::lock_guard lk(ring_mu);
    uint64_t at = mesh_emu_region_at(ring_cursor, bytes, usable_ring);
    if (at == kMeshEmuNoRoom) {
        at = 0;   // (the ring wraps; the caller keeps the frame's regions inside one pass of it)
    }
    ring_cursor = at + bytes;
    region.offset = at;
    region.index_at = mesh_emu_index_at(at, vertices);
    region.vertices = vertices;
    region.primitives = primitives;
    return region;
}

VkDescriptorSetLayout mesh_emu_set_layout() {
    static VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (layout || !ring) {
        return layout;
    }
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1,
                                         VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 1;
    ci.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(dev.device, &ci, nullptr, &layout) != VK_SUCCESS) {
        layout = VK_NULL_HANDLE;
    }
    return layout;
}

VkDescriptorSet mesh_emu_set() {
    static VkDescriptorSet set = VK_NULL_HANDLE;
    if (set || !ring) {
        return set;
    }
    const VkDescriptorSetLayout layout = mesh_emu_set_layout();
    if (!layout) {
        return VK_NULL_HANDLE;
    }
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(dev.device, &pci, nullptr, &pool) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(dev.device, &ai, &set) != VK_SUCCESS) {
        set = VK_NULL_HANDLE;
        return set;
    }
    VkDescriptorBufferInfo info{ring, 0, kMeshEmuMaxRegionBytes};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(dev.device, 1, &write, 0, nullptr);
    return set;
}

void mesh_emu_bind(VkCommandBuffer cmd, VkPipelineLayout layout, VkPipelineBindPoint bind_point,
                   uint64_t region_offset) {
    const VkDescriptorSet set = mesh_emu_set();
    if (!set) {
        return;
    }
    const auto offset = static_cast<uint32_t>(region_offset);
    vkCmdBindDescriptorSets(cmd, bind_point, layout, 1, 1, &set, 1, &offset);
}

void mesh_emu_clear(VkCommandBuffer cmd, const MeshEmuRegion& region, uint32_t offset, uint32_t size,
                    uint32_t first_task) {
    if (ring && size != 0) {
        vkCmdFillBuffer(cmd, ring, region.offset + offset, size, 0);
        if (first_task != 0 && offset == 0 && size >= 12) {
            vkCmdUpdateBuffer(cmd, ring, region.offset + 8, sizeof(uint32_t), &first_task);
        }
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                           VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    }
}

void mesh_emu_dispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                       const MeshEmuRegion& region, uint32_t groups) {
    if (!ring || !pipeline || groups == 0) {
        return;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    mesh_emu_bind(cmd, layout, VK_PIPELINE_BIND_POINT_COMPUTE, region.offset);
    vkCmdDispatch(cmd, groups, 1, 1);
}

void mesh_emu_task_dispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                            const MeshEmuRegion& region, uint32_t x, uint32_t y, uint32_t z) {
    if (!ring || !pipeline || x == 0 || y == 0 || z == 0) {
        return;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    mesh_emu_bind(cmd, layout, VK_PIPELINE_BIND_POINT_COMPUTE, region.offset);
    vkCmdDispatch(cmd, x, y, z);
}

void mesh_emu_dispatch_indirect(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                                const MeshEmuRegion& region) {
    if (!ring || !pipeline) {
        return;
    }
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                       VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    mesh_emu_bind(cmd, layout, VK_PIPELINE_BIND_POINT_COMPUTE, region.offset);
    vkCmdDispatchIndirect(cmd, ring, region.offset + kMeshEmuIndirectAt);
}

void mesh_emu_finalize(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                       const MeshEmuRegion& region) {
    if (!ring || !pipeline) {
        return;
    }
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    mesh_emu_bind(cmd, layout, VK_PIPELINE_BIND_POINT_COMPUTE, region.offset);
    vkCmdDispatch(cmd, 1, 1, 1);
}

void mesh_emu_draw_indirect(VkCommandBuffer cmd, const MeshEmuRegion& region, VkPipeline pipeline) {
    if (getenv("RX_VK_MESH_EMU_DUMP")) {
        // (diagnostic: the previous draw's region is complete a frame later; read what the compute wrote)
        static MeshEmuRegion previous{};
        static uint64_t prev_frame = 0;
        static bool has_previous = false;
        static int dumped = 0;
        const uint64_t current_frame = texture_frame_no();
        if (has_previous && ring_map && current_frame > prev_frame && dumped < 64) {
            dumped++;
            vkDeviceWaitIdle(dev.device);
            const uint8_t* base = static_cast<const uint8_t*>(ring_map) + previous.offset;
            uint32_t header[9]{};
            std::memcpy(header, base, sizeof header);
            float p0[4]{}, p1[4]{};
            std::memcpy(p0, base + 64, sizeof p0);
            std::memcpy(p1, base + 80, sizeof p1);
            uint32_t idx[6]{};
            std::memcpy(idx, static_cast<const uint8_t*>(ring_map) + previous.index_at, sizeof idx);
            logf("vk: mesh emu ring %#llx (frame %llu): vertex cursor %u index cursor %u; indirect %u %u %u %u %u; "
                 "p0 %.2f %.2f %.2f %.2f; p1 %.2f %.2f %.2f %.2f; i %u %u %u %u %u %u",
                 (unsigned long long)previous.offset, (unsigned long long)prev_frame,
                 header[0], header[1], header[4], header[5],
                 header[6], header[7], header[8], p0[0], p0[1], p0[2], p0[3], p1[0], p1[1], p1[2],
                 p1[3], idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
            has_previous = false;
        }
        if (!has_previous) {
            previous = region;
            prev_frame = current_frame;
            has_previous = true;
        }
    }
    if (!ring || !pipeline || region.vertices == 0) {
        return;
    }
    // (the compute writes were made visible by mesh_emu_compute_done() before the render pass began)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    // (the index list follows the vertex records in the region; the indirect command's firstIndex is relative to it)
    vkCmdBindIndexBuffer(cmd, ring, region.index_at, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(cmd, ring, region.offset + kMeshEmuIndirectAt, 1,
                             (uint32_t)sizeof(VkDrawIndexedIndirectCommand));
}

void mesh_emu_compute_begin(VkCommandBuffer cmd) {
    if (!ring) {
        return;
    }
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_NONE;   // (write-after-read: an execution dependency is enough)
    mb.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_NONE;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

void mesh_emu_compute_done(VkCommandBuffer cmd) {
    if (!ring) {
        return;
    }
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                       VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

} // namespace mkwvk
