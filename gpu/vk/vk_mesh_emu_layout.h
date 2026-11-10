// Layout of one draw's region in the mesh-emulation ring (Phase 3, docs/arm64-phase3.md).
//
// Pure integer arithmetic, no Vulkan: the ring allocation and its wrap are unit-tested offline
// (tests/test_mesh_emu_layout.py) while the Vulkan half lives in vk_mesh_emu.cpp.
#pragma once

#include <cstdint>

namespace mkwvk {

inline constexpr uint64_t kMeshEmuRingBytes = 256ull << 20;   // (the persistent ring the design asks for)
inline constexpr uint64_t kMeshEmuNoRoom = ~uint64_t{0};
inline constexpr uint32_t kMeshEmuAlign = 256;

// One region:
//   +0   u32 vertexCount, +4 u32 primitiveCount, +8 u32 firstVertex, +12 u32 firstIndex
//   +16  VkDrawIndexedIndirectCommand (20 bytes)
//   +64  vertex records (kMeshEmuVertexStride each: the clip position vec4)
//   ...  index list (kMeshEmuIndexStride each, 3 a primitive)
inline constexpr uint32_t kMeshEmuHeaderBytes = 64;
inline constexpr uint32_t kMeshEmuIndirectAt = 16;
inline constexpr uint32_t kMeshEmuVertexStride = 16;
inline constexpr uint32_t kMeshEmuIndexStride = 4;

inline uint64_t mesh_emu_region_bytes(uint32_t vertices, uint32_t primitives) {
    return kMeshEmuHeaderBytes + uint64_t{vertices} * kMeshEmuVertexStride +
           uint64_t{primitives} * 3 * kMeshEmuIndexStride;
}

inline uint64_t mesh_emu_align(uint64_t n) {
    return (n + kMeshEmuAlign - 1) & ~uint64_t{kMeshEmuAlign - 1};
}

/// Where a region of `bytes` starts: `cursor` (aligned up) when it fits before the end, else 0 (the ring wraps).
/// kMeshEmuNoRoom when it cannot fit in the ring at all.
inline uint64_t mesh_emu_region_at(uint64_t cursor, uint64_t bytes, uint64_t ring = kMeshEmuRingBytes) {
    if (bytes == 0 || bytes > ring) {
        return kMeshEmuNoRoom;
    }
    const uint64_t at = mesh_emu_align(cursor);
    return at + bytes <= ring ? at : 0;
}

/// Byte offset of the region's index list (what the index buffer binding starts at).
inline uint64_t mesh_emu_index_at(uint64_t region_offset, uint32_t vertices) {
    return region_offset + kMeshEmuHeaderBytes + uint64_t{vertices} * kMeshEmuVertexStride;
}

} // namespace mkwvk
