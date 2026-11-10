# Phase 2 on the Snapdragon X Elite: what the Adreno can do, and what it means

Measured on the Surface Pro 11 (`vulkaninfo` 1.3.301, driver 1.3.295), 2026-11-09. This replaces the assumptions the
original Phase 2 brief was written against; two of them are wrong for this GPU.

## 1. Measured capabilities (Adreno X1-85)

| Capability | Value | Consequence |
|---|---|---|
| `VK_EXT_mesh_shader` | **no** | Phase 3 is mandatory, not optional |
| `VK_EXT_external_memory_host` | **no** | Phase 2 is mandatory; guest pools cannot be imported |
| `VK_KHR_external_memory_win32` | **no** | no section-handle route either |
| `textureCompressionASTC_LDR` | **yes** | native ASTC already works — Phase 2.2 needs no decoder bypass |
| `textureCompressionBC` | **yes** | the BC3 keeping path (`vk_texture.cpp`) works |
| `bufferDeviceAddress` | yes | pool buffers can keep using device addresses |
| `descriptorIndexing`, `timelineSemaphore`, `dynamicRendering`, `synchronization2` | yes | the backend's assumptions hold |
| `maxMemoryAllocationSize` | 1 GiB | the existing 16 MB import chunking stays |
| `maxBufferSize` | ~4 GiB | not the limit |

Heaps and memory types:

```
heap 0: 7.80 GiB DEVICE_LOCAL
heap 1: 4.00 GiB DEVICE_LOCAL
type 3: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
type 4: DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
```

Type 3/4 is the type the brief asked for, and it exists. It is the enabler for the design below.

## 2. The brief's Phase 2.1 as written cannot work here

It says: *map guest RAM directly via Vulkan … and hand the base pointer to the CPU runtime*.

Handing a Vulkan allocation's base pointer to the guest runtime only works if that pointer **is** the guest address:
the runtime addresses guest memory as `A64_MEM + guest_addr`, and the game's NVN pools get their guest addresses
from the guest's own allocator. A Vulkan allocation has a driver-chosen host pointer, and there is no Vulkan call that
maps it at a chosen virtual address. The only mechanism that makes the two coincide is importing the guest pages into
Vulkan — and that needs `VK_EXT_external_memory_host` (or a win32 section, `VK_KHR_external_memory_win32`), **neither
of which this device exposes**.

So on this GPU, "zero copy" in the brief's sense is impossible. This is a property of the driver, not of Hookshot.

## 3. What is achievable: the UMA copy pool

Each pool becomes a `VkBuffer` allocated from memory type 3/4, mapped, with a device address as today:

- **import:** `memcpy(map, guest + lo, bytes)` once.
- **CPU → GPU:** at each submission, `memcpy` the *dirty* 4 KB pages from guest memory into `map`. The page
  generations (`page_gen`, `region_gen`, `A64_WRITE_TRACK`) already exist for the mirror path.
- **GPU → CPU:** after the submission's fence, `memcpy` back the ranges the GPU wrote — the ranges the existing
  `writebacks` vector already computes.
- **no separate mirror:** bindings already use `p.buffer` when `mirror_on()` is false, so the pool buffer *is* the
  binding target. No third copy.

This is "no staging buffer", not "no copy": exactly one copy per changed range, RAM→RAM on UMA (no PCIe).

### Code sites (all in `gpu/vk/vk_memory.cpp` unless noted)

| Site | Change |
|---|---|
| `memory_init` :525 | `copy = !dev.external_memory_host`; query `Vulkan11/13Properties` without the host-pointer struct; `align = 4096`; log the mode instead of `return false` |
| `Pool` :45 | add `void* map`, `bool copy` |
| `import_pool` :316 | branch to a new `pool_alloc_copy()` before the `VkExternalMemoryBufferCreateInfo` path |
| `mirror_on` :1820 | return false in copy mode |
| `mirror_sync` :1962 | copy-mode branch: the same `runs`, `memcpy` into `p.map` instead of `vkCmdCopyBuffer` |
| write-back path :1900–1957 | copy-mode `memcpy` back to guest at completion |
| `vk_replay.cpp` :3485 | unchanged (its `memory_init()` call now succeeds) |
| `vk_main.cpp` | unchanged: the extension is already `optional(...)` |

### Risks

1. **Memory, not just speed.** An imported pool costs no extra RAM; a copy pool holds a second full copy. On a 16 GB
   UMA machine running a ~10 GB game this is the central risk. Mitigations: copy only pools actually bound as
   buffers, keep a budget like `mirror_budget()`, give pools back when idle, and rely on native ASTC (below) to
   reclaim the ~3.5 GB the decoder buffers would otherwise take.
2. **Coherence.** Must pick a `HOST_COHERENT` type (3 or 4); type 4 (`HOST_CACHED`) is better for readback.
3. **Ordering.** Host writes to coherent memory are visible to the GPU after queue submit; no flush needed, but the
   readback direction needs the submission's completion fence (already tracked as `mem_completed`).
4. **The 1 GiB allocation limit** keeps the existing 16 MB chunking.

## 4. Phase 2.2 is already satisfied

`gpu/vk/vk_formats.cpp:29` uploads native `VK_FORMAT_ASTC_*_UNORM_BLOCK` when `textureCompressionASTC_LDR` is set,
and the Adreno reports it. `vk_texture.cpp`'s compute decoder is the fallback, and its BC3 keeping path is also
available (`textureCompressionBC`). **No code change is needed for native ASTC on this device** — the remaining work
is to confirm at runtime that no pool forces the decoder (`RX_VK_ASTC_CPU`) and to measure the real footprint.

## 5. What this changes in the plan

- Phase 2 is a **"make it run"** task on this device, not a "save 3.5 GB" task. The saving comes from ASTC already.
- Phase 3 (mesh → compute + indirect draw) remains the larger piece and is unaffected by these findings.
- Verification needs the game dump (`OOT_DUMP`) and the CPM cache for the Vulkan headers; neither is in the
  workspace yet, so the copy mode below is compile-verified only (section 6).

## 6. Status

### Implemented (compile-verified, not yet run)

Increment 1, the copy-pool allocation and both copy directions:

- `gpu/vk/vk_loader.h`: `vkUnmapMemory` loaded (the copy pools are mapped).
- `gpu/vk/vk_memory.cpp`:
  - `copy_pools` mode; `memory_init()` no longer returns false without `VK_EXT_external_memory_host`; it queries the
    Vulkan 1.1/1.3 properties in both cases and reports the mode.
  - `import_pool_copy()`: buffer in `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (prefers `HOST_CACHED`), device
    address as today, mapped.
  - `copy_sync()`: the whole range at a pool's first bind, then only the pages the CPU wrote, using the same
    `page_gen`/`region_gen` generations as the mirror path.
  - `gpu_buffer_bind()`: copy mode syncs at the first bind of a submission (the lock-free pool cache is skipped
    there, because that path cannot copy), and records every `written` binding.
  - `copy_flush_locked()` / `copy_wb_pending`: the GPU's writes are copied back into guest memory once the
    submission that wrote them is complete. The completion is known from `completed()` when the next submission
    starts (`memory_take_capture`, `memory_capture_now`, `memory_begin_submission`), so the flush never blocks. It
    runs **before** anything is given back, so the mapped pool is still alive.
  - Budget: `copy_budget()` (RX_VK_COPY_MB, else two thirds of the largest device-local heap — 5.2 GiB here) and
    `copy_live_bytes()`; `imports_give_back()` evicts copy pools idle for `kCopyIdle` submissions once over budget.
    The existing system-heap path cannot do this: it sums only non-device-local heaps, and this device has none.
  - Lifetime: retired pools are given back in copy mode and unmapped before they are freed.

Verification: `clang -fsyntax-only --target=aarch64-w64-mingw32` and `--target=x86_64-w64-mingw32` on
`vk_memory.cpp`, `vk_main.cpp` and `vk_texture.cpp` all exit 0. The x64 path is untouched: `copy_pools` is false
whenever `VK_EXT_external_memory_host` is present.

### Known limitation

The GPU → guest copy is one submission late at worst. The mirror path writes back inside the submission's own
command buffer, so guest memory is correct the moment the GPU is done; a CPU-side copy cannot be recorded there.
For OoT this may show as a one-frame-stale exposure meter. If it is visible, the fix is a small fence-triggered flush
on the replay thread, not a design change.

### What is still open

1. **Runtime verification.** Needs the dump and the CPM cache; without them the copy mode has never run.
2. Whether every pool that is bound fits the 5.2 GiB budget; the first run's `vk: copy pools:` lines will say.
3. Phase 3 (mesh → compute + indirect draw) is untouched.


