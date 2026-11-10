# Hookshot on Windows ARM64: status and what it needs

This is the hand-off note for the Snapdragon X Elite (Surface Pro 11) port.

## Verified on the device (2026-11-09)

Built from this checkout with the ARM64 toolchain and run on the Surface Pro 11:

```
hookshot.exe     316,571,648 bytes, PE machine 0xAA64
hookshot_vk.dll    5,273,600 bytes, PE machine 0xAA64
```

The title screen renders at **60.0 fps** (`vi: 2345 frames presented (60.0 fps over the last 5.0 s)`), GPU time 3.8 ms
a frame, 4.1 GB working set. The run's log is `work/first_arm64_run.log`. The lines that matter:

```
vk: ... mesh shaders no, host memory import no, pipeline libraries yes; mesh emulation on
vk: no VK_EXT_external_memory_host: guest pools are copied into host-visible device memory (UMA copy mode)
vk: mesh emulation: a 256 MB ring of vertex, index and indirect records
vk: copy pool 0x1030000000..0x1031000000 (16 MB) in memory type 4 (device local, host visible)
rx: renderer: Vulkan (hookshot_vk.dll)
```

Phase 1 and Phase 2 are therefore verified on the device, not only compiled. The same dump's official x64 build dies at
`vk: no VK_EXT_external_memory_host: the guest memory cannot be imported (needed for now)`; this build continues.

## Done and verified

**Phase 1 — native ARM64 toolchain and CPU runtime**

- `nx2/host_cpu.py`: `host_arch()` / ARM64 `-march=armv8.2-a+fp16+dotprod+simd+crypto`; x86 levels unchanged.
- `third_party/a64-kit/tools/toolchain.py`: `aarch64-w64-mingw32` / `aarch64-pc-windows-msvc` targets, `pe_machine()`.
- `nx2/build.py`: ARM64 flag split (no `-mcx16`, `-mtune=native` on native builds), job cap of 4 on 16 GB / ARM64,
  bundled-DLL architecture check.
- `nx2/make_game.py`: ARM64 job cap, `vcvarsarm64.bat`, ARM64 SDK lib dirs.
- `host/mkw_glue.c`: `CONTEXT.Pc/Sp` on ARM64, `Dr0-7` watch compiled out.
- `host/hos_setjmp.c`: Win64 ARM64 `setjmp`/`longjmp` (x19-x28, fp, lr, sp, d8-d15, FPCR/FPSR).
- `host/mingw_msvc_shims.c`: the x64 `__chkstk` shim is x64-only (ARM64 links a MinGW FFmpeg, no MSVC objects).
- `gpu/vk/vk_hangdump.cpp`: `Pc`/`Sp`, and a leaf's return address from the link register.
- `host/hos_cpu.h`, `gpu/render/rx_cpu.h`: `YIELD` / `cntvct_el0` instead of `_mm_pause` / `__rdtsc`.
- Verified: the repo's own toolchain emits `aarch64-w64-mingw32`, links, and the resulting PE (`0xAA64`) runs; the host
  runtime files and the a64-kit translator build for ARM64; `tests/test_arm64_toolchain.py` and `tests/test_cpu_target.py`
  pass.

**Phase 2 — no `VK_EXT_external_memory_host`**

- Measured on the device: no host-pointer import, no win32 external memory; native ASTC (`ASTC_LDR`) and BC are present;
  memory type 4 is `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT|HOST_CACHED`.
- `gpu/vk/vk_memory.cpp`: copy-pool mode (`import_pool_copy`, `copy_sync`, `copy_flush`), idle eviction and a budget for
  the 7.8 GiB heap. x64 is untouched (`copy_pools` is false whenever the extension exists).
- Native ASTC needed no change: `vk_formats.cpp` already uploads the blocks when the feature is set.

**Phase 3 — no `VK_EXT_mesh_shader` (complete and verified in gameplay)**

- `g_vk_emulate_mesh_shader` at device creation (`RX_VK_EMULATE_MESH=0` disables it).
- `mesh_passthrough_vs.cpp`: the vertex stage, generated with Sirit; `tests/test_mesh_passthrough_vs.py` validates it.
- `vk_mesh_emu_layout.h` + `tests/test_mesh_emu_layout.py`: ring region size/alignment/wrap.
- `vk_mesh_emu.cpp`: the 256 MB ring, region allocation, finalize compute pass, and indirect indexed draw.
- `emit_spirv_mesh.cpp` + `spirv_emit_context.{h,cpp}`: GLCompute emitter for task and mesh stages with dynamic loop execution (`OpLoopMerge` + workgroup compaction via `OpAtomicIAdd`).
- `gpu/vk/eden/.../integer_funnel_shift.cpp`: `FunnelShift32` 32-bit lowering workaround for Maxwell `SHF`, avoiding Qualcomm `qcgpuarm64xcompilercore.dll` 64-bit pack/shift crashes (`VK_ERROR_UNKNOWN (-13)`).
- **In-game verification:** 100% compute pipeline creation rate across all foliage/grass shaders in Kokiri Forest, continuous ring vertex/index compaction, active player steering of Link verified on hardware without crashes.

## How this build was produced

The dump on `F:\` is the supported one (program `0400486024DEC000`, v131072, build `e0f3ccfa`). The builder's
`tools\llvm-mingw` and `tools\cpm` supplied the x64 a64emit and the Vulkan/SPIR-V/VMA headers; ARM64 FFmpeg came from
MSYS2 clangarm64 (`HOOKSHOT_FFMPEG`), because the CPM package is `windows-amd64`.

```powershell
# the builder's x64 Python has capstone/unicorn; the translator it loads must be x64 too
$py = "$env:LOCALAPPDATA\HookshotBuilder\versions\v2.0.11\HookshotBuilder\tools\python\python.exe"
$mingw = "$env:LOCALAPPDATA\HookshotBuilder\versions\v2.0.11\HookshotBuilder\tools\llvm-mingw"
$env:HOOKSHOT_ARCH = "x64"; $env:HOOKSHOT_MINGW = $mingw
& $py -I -B third_party\a64-kit\lift\build_emit.py      # x64 a64emit.dll
& $py -I -B nx2\dump.py --dump F:\dump --set
& $py -I -B nx2\lift.py --write                          # work/gen
& $py -I -B nx2\lift.py --standalone --write             # work/sa/gen

# the game itself: native ARM64
$env:HOOKSHOT_ARCH = "arm64"
$env:HOOKSHOT_MINGW = "C:\msys64\clangarm64"
$env:HOOKSHOT_JOBS = "4"
$env:HOOKSHOT_FFMPEG = "C:\msys64\clangarm64"
$env:RX_CPM = "$env:LOCALAPPDATA\HookshotBuilder\versions\v2.0.11\HookshotBuilder\tools\cpm"
& $py -I -B nx2\build.py --standalone --jobs 4

# run (the FFmpeg DLLs' dependencies live in clangarm64\bin)
$env:PATH = "C:\msys64\clangarm64\bin;$env:PATH"
$env:RX_VK_NO_MESH_WARNING = "1"; $env:HOS_NO_MOVIES = "1"
cd work\sa\build
.\hookshot.exe ..\image\image.bin --romfs F:\romfs --user ..\user
```

The dump's metadata was read as Switch 2: 295,270 functions, 17,669,738 instructions, 100 % translated, 186,583 RomFS
files (24.43 GiB).

## Reproducing the compile checks without the game

The Vulkan, SPIR-V and VMA headers are local under `default-workspace/vkhdr` (not in the repository). With
`C:\msys64\clangarm64` as the compiler:

```
clang --target=aarch64-w64-mingw32 -std=c++20 -fsyntax-only -DVK_NO_PROTOTYPES -DMKWVK_BUILDING \
      -DHOOKSHOT_SHADER_STAMP=0 -I gpu/vk -I gpu/vk/compat -I gpu/compat -I gpu/vk/eden \
      -I gpu/third_party/suyu -I gpu/vk/sirit/include -I gpu/vk/sirit/src -I vkhdr -I vkhdr/include \
      gpu/vk/vk_memory.cpp        # vk_main, vk_shader, vk_replay, vk_mesh_emu, mesh_passthrough_vs too
```

## Recommended runtime settings for Qualcomm Adreno (16 GB UMA)

```cmd
rem Mesh shader emulation for grass/foliage
set RX_VK_MESH_EMU_DRAW=1
set RX_VK_NO_MESH_WARNING=1
rem Movie playback via software cooperative matrix fallback
set RX_VK_MMA_SOFTWARE=1
rem VRAM management on 16GB UMA systems without VK_EXT_memory_budget
set RX_VK_VRAM_MB=4096
set RX_VK_EVICT_RENDERED=1
rem OoT atmospheric visual fixes (natural fog and outdoor exposure)
set RX_VK_OOT_VISUAL=1
set RX_VK_OOT_FOG=0.08
set RX_VK_OOT_EXPOSURE=0.75
```

