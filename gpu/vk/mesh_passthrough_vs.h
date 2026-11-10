// The pass-through vertex shader of the mesh-shader emulation (Phase 3, Windows on ARM).
//
// A device without VK_EXT_mesh_shader runs the translated mesh program as a compute dispatch, which appends vertex
// records and primitives to a ring buffer (see docs/arm64-phase3.md). This shader is the draw's vertex stage: it reads
// one record per vertex from that buffer and writes the clip position, so the graphics pipeline is an ordinary
// indexed indirect draw over the compute-written data.
//
// It is generated with Sirit rather than shipped as a binary blob, so it needs no shader compiler at build time and is
// unit-testable offline (tests/test_mesh_passthrough_vs.py).
#pragma once

#include <cstdint>
#include <vector>

namespace mkwvk {

/// SPIR-V words of the pass-through vertex shader. Descriptor set 1, binding 0 is a storage buffer of vertex records;
/// the first field of a record is the clip-space position (vec4).
std::vector<uint32_t> mesh_passthrough_vs();

/// SPIR-V words of the finalize pass (one invocation): reads the ring's primitive counter, validates and packs padded index slots and writes the
/// VkDrawIndexedIndirectCommand (5 words) at `indirect_word`. Descriptor set 1, binding 0 is the ring.
std::vector<uint32_t> mesh_finalize_spv(uint32_t counter_word, uint32_t indirect_word,
                                         uint32_t max_vertices, uint32_t max_primitives,
                                         uint32_t corners);

} // namespace mkwvk
