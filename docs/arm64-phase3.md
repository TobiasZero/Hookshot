# Phase 3: mesh shaders without VK_EXT_mesh_shader

The Adreno X1-85 does not expose `VK_EXT_mesh_shader` (measured, see `docs/arm64-phase2.md`), and OoT draws its
foliage, grass and several effects with NVN2 task/mesh programs. Phase 3 runs those programs as a compute dispatch and
an ordinary indexed indirect draw.

## How the mesh path works today

Eden's backend emits a real Vulkan mesh module:

- `emit_spirv.cpp:278-284` — `Stage::Task`/`Stage::Mesh` map to `spv::ExecutionModel::MeshEXT`.
- `emit_spirv_mesh.cpp` — the NVN stage buffer stays in workgroup memory; `MeshEpilogue` copies it into the mesh output
  builtins (`gl_MeshVerticesEXT`, `gl_PrimitiveTriangleIndicesEXT`, `gl_MeshPrimitivesEXT`).
- `emit_spirv.cpp:610-640` — the epilogue leaves markers that are patched into `OpSetMeshOutputsEXT` and
  `OpEmitMeshTasksEXT`.
- `vk_draw.cpp:4779-4902` — binds the task/mesh stages and records `vkCmdDrawMeshTasks*EXT`.

The output variables are the only place the program's results leave the shader. That is what makes the rewrite
tractable: the same epilogue can write to a storage buffer instead.

## Design

### Data layout (one region per draw, in a 256 MB ring buffer)

```
+0    u32 vertexCount          (written by the compute epilogue)
+4    u32 primitiveCount
+8    u32 firstVertex          (the region's base, for the pass-through shader)
+12   u32 firstIndex
+16   indirect command (VkDrawIndexedIndirectCommand, 20 bytes)
+36   pad to 64
+64   vertex records: vertexCount x 16 bytes (position vec4 today; varietyings are appended here later)
...   index list: primitiveCount x 3 x u32
```

The ring buffer is allocated from a `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` memory type (type 4 on this device), the
same choice the Phase 2 copy pools make, so it is CPU-initialised and GPU-read/written without a staging buffer.

### Shader translation (`EmitContext::mesh_emu`)

1. `emit_spirv.cpp`: when `mesh_emu`, emit `ExecutionModelGLCompute`; a mesh stage gets
   `LocalSize(mesh_threads, 1, 1)`; a task stage keeps its own local size.
2. `emit_spirv_mesh.cpp` `MeshEpilogue`:
   - replace the `OpSetMeshOutputsEXT` marker with `OpAtomicIAdd` on the region's vertex counter, then the same for
     primitives (the counts are already computed there);
   - replace `ctx.mesh_vertex_targets` writes (`CopyLines`) with stores into the vertex-record SSBO at
     `firstVertex + index`;
   - replace `ctx.mesh_primitive_indices` with stores into the index list;
   - write the indirect command's `indexCount`, `firstIndex`, `vertexOffset`.
3. `TaskEpilogue`: replace `OpEmitMeshTasksEXT` with an `OpAtomicIAdd` that reserves the mesh-workgroup range, and
   write the payload blob the mesh stage reads (the payload is already a workgroup array).
4. The task/mesh pair stays two dispatches, recorded one after the other with a `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`
   barrier between them.

The exact per-element index math already exists in `CopyLines` (`first = base + (index/32)*lines*32 + index%32`), so
the SSBO store replaces the builtin store one-to-one.

### Runtime (`gpu/vk/vk_mesh_emu.cpp`)

- `mesh_emu_init()`: ring buffer, indirect buffer, descriptor set layout for the vertex/index SSBOs, module from
  `mesh_passthrough_vs()`.
- `mesh_emu_begin_frame(cmd)`: bump the ring's frame region, reset the cursor.
- `mesh_emu_draw(cmd, task_pipeline, mesh_pipeline, graphics_pipeline, draw)`:
  1. `vkCmdBindPipeline(COMPUTE)` + descriptor sets, `vkCmdDispatch` (task, if any), barrier;
  2. `vkCmdDispatch` (mesh), barrier `COMPUTE_SHADER -> INDIRECT_COMMAND_READ|VERTEX_ATTRIBUTE_READ|INDEX_READ`;
  3. `vkCmdBindPipeline(GRAPHICS)` + the pass-through vertex input (none) and the original fragment stage,
     `vkCmdDrawIndexedIndirect(cmd, indirect, offset, 1, 20)`.

### Where the pieces connect

| File | Change |
|---|---|
| `eden/.../spirv/emit_spirv.cpp` | execution model + marker patch |
| `eden/.../spirv/emit_spirv_mesh.cpp` | epilogue writes to SSBOs |
| `eden/.../spirv/spirv_emit_context.{h,cpp}` | `mesh_emu` member, SSBO variables, bindings |
| `vk_shader.cpp:1442` | `program_for(Mesh)` no longer returns null when `g_vk_emulate_mesh_shader` |
| `vk_draw.cpp:4779-4902` | the mesh branch calls `mesh_emu_draw` instead of `vkCmdDrawMeshTasks*` |
| `vk_main.cpp` | `g_vk_emulate_mesh_shader` (done) |
| `vk_mesh_emu.cpp` | new: ring, pipelines, indirect draw |
| `mesh_passthrough_vs.cpp` | done: the vertex stage |

## Status

- `g_vk_emulate_mesh_shader`: done (`vk_main.cpp`), on when the device lacks the extension, `RX_VK_EMULATE_MESH=0`
  disables it.
- `mesh_passthrough_vs.cpp`: done; `tests/test_mesh_passthrough_vs.py` builds it with Sirit on the host and checks the
  module (SPIR-V 1.3, one `Vertex` entry point `main`, storage-buffer load into `gl_Position`).
- `vk_mesh_emu_layout.h`: done; `tests/test_mesh_emu_layout.py` checks region size, 256-byte alignment, wrap and
  "no room" (which is distinct from a valid region at offset 0).
- `vk_mesh_emu.cpp`: done, the ring and the draw half. A 256 MB ring (VMA, mapped, device-local on this UMA part),
  `mesh_emu_alloc` hands out regions, `mesh_emu_draw_indirect` records the compute→indirect/vertex/index barrier,
  binds the ring as the index buffer and issues one `vkCmdDrawIndexedIndirect`. `replay_start` calls
  `mesh_emu_init()` after `vma_init`; on a device with mesh shaders it is a no-op.
- **Emitter (shader side): done.** With `Shader::Backend::SPIRV::g_mesh_emu` set, a mesh program is emitted as
  `GLCompute`; the stage buffer stays in workgroup memory, and positions and triangle indices go to a storage buffer
  (`mesh_outputs`: `vec4 positions[]`, `uvec3 indices[]` with 16-byte strides, and a `u32 counters[4]` member). Each
  workgroup takes one vertex range and one index range from the counters with `OpAtomicIAdd` (Device scope), so
  several workgroups compact into one list; the indices are rebased to the compacted vertices. Validated against a
  real OoT mesh program: `work/shadertool_emu.exe work/mesh_stage.code work/mesh_emu.spv --stage mesh --emu`, then
  `spirv-val --target-env vulkan1.3 --uniform-buffer-standard-layout` exits 0, entry point `GLCompute`.
- **Ring binding:** `vk_mesh_emu` creates set 1 (binding 0, a dynamic storage buffer over the 256 MB ring) and
  `mesh_emu_bind()` binds it with the region offset; `ensure_layout` adds that set, and the compute/vertex stages, to
  an emulated mesh program's pipeline layout. The title screen still runs at 60 fps with this in place.
- **Remaining (runtime side):** build the compute pipeline from the emulated module, a graphics pipeline with the
  pass-through vertex shader plus the original fragment stage, a 1-thread finalize pass that turns the index counter
  into the indirect command, and the draw call (`mesh_emu_draw_indirect`); then set `g_mesh_emu` in `vk_shader.cpp`,
  return the emulated program from `mesh_program_locked`, and test gameplay. The emitter supports positions and
  triangle indices only; generic outputs, per-primitive outputs and task stages throw, which covers the sampled OoT
  foliage shader but not necessarily all 500.
- `vk_shader.cpp`: a run on a device without mesh shaders now says once that the task/mesh programs are not translated to
  compute yet, instead of silently returning no program. The game still runs with the path disabled (60 fps on the
  title, foliage missing).

## Offline corpus and translator (for developing the emitter)

The builder's shader seed (`F:\hookshot_shaders.bin`, `HKSEED01`) holds 42,856 stages: 22,356 vertex, 15,852
header-only, 4,128 compute, **500 mesh and 20 task**, in 1,312 programs with a task or mesh stage. Two are extracted:

```
work/task_stage.code   10,496 bytes
work/mesh_stage.code   30,976 bytes
```

`work/parse_seed.py` reads the seed (header, stages, programs) and writes the first task/mesh pair. `gpu/shadertool.cpp`
now takes a stage for the programs whose header does not name it, and can be linked from the objects the game build
already produced:

```powershell
clang++ --target=aarch64-w64-mingw32 -std=c++20 -O1 -fms-extensions -D_CRT_SECURE_NO_WARNINGS -Wno-everything ^
  -Igpu/vk/eden -Igpu/third_party/suyu -Igpu/vk/compat -Igpu/compat -Igpu/vk/sirit/include -Igpu/vk/sirit/src ^
  -I..\vkhdr\include -c gpu/shadertool.cpp -o work/shadertool.o
# link work/shadertool.o + work/sa/build/obj/vk_eden_*.o + vk_sirit_*.o
work\shadertool.exe work\mesh_stage.code work\mesh.spv --stage mesh
```

Result: `work/mesh.spv`, a complete `MeshEXT` module (93,316 words, 64 vertices, 126 triangles, `OpSetMeshOutputsEXT`,
`PrimitiveTriangleIndicesEXT`), and `work/task.spv` (`TaskEXT`). It validates:

```
spirv-val --target-env vulkan1.3 --uniform-buffer-standard-layout work/mesh.spv   # exits 0
```

These are the inputs the new emitter mode is developed and validated against; no GPU or gameplay is needed.

## Risks

1. **Draw ordering.** Each draw needs its own region; the ring must not wrap while an earlier draw of the same
   submission still reads it. Size the ring by frames in flight, not by draws.
2. **Counts and clamping.** The existing epilogue clamps indices and counts (`Clamped`, `task_cap`); the SSBO path must
   keep every clamp — an unwritten byte used to draw garbage geometry on NVIDIA.
3. **Topology.** Triangles, lines and points all use the same path; the index list for a point is one index, not three.
4. **16-bit indices.** The index list is 32-bit; NVN's 8/16-bit index bytes are widened in the epilogue, where they are
   already read.
