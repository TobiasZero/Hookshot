"""The mesh-emulation ring layout: region size, alignment and wrap (gpu/vk/vk_mesh_emu_layout.h).

Pure integer arithmetic, so this needs no GPU and no game files: a C++ fixture includes the header and checks the
cases the runtime depends on.

python -I -B tests/test_mesh_emu_layout.py [--cxx clang++]
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include "vk_mesh_emu_layout.h"
#include <cstdio>

using namespace mkwvk;

static int failures = 0;
static void eq(const char* what, unsigned long long got, unsigned long long want) {
    if (got != want) {
        std::printf("FAIL %s: got %llu, want %llu\n", what, got, want);
        ++failures;
    }
}

int main() {
    // (64-byte header + 4 vertices x 16 + 2 primitives x 3 x 4)
    eq("region bytes", mesh_emu_region_bytes(4, 2), 64 + 4 * 16 + 2 * 3 * 4);
    eq("region bytes empty", mesh_emu_region_bytes(0, 0), 64);
    eq("align(1)", mesh_emu_align(1), 256);
    eq("align(256)", mesh_emu_align(256), 256);
    eq("align(257)", mesh_emu_align(257), 512);
    eq("first region", mesh_emu_region_at(0, 152), 0);
    eq("aligned up", mesh_emu_region_at(1, 152), 256);
    // (the ring wraps when the region would pass its end; 0 is the start, which is a valid region)
    eq("wrap", mesh_emu_region_at(kMeshEmuRingBytes - 128, 152), 0);
    // (exactly to the end still fits)
    eq("fits to end", mesh_emu_region_at(kMeshEmuRingBytes - 256, 256), kMeshEmuRingBytes - 256);
    // (a region bigger than the ring: no room, which is not the same as offset 0)
    eq("no room", mesh_emu_region_at(0, kMeshEmuRingBytes + 1), kMeshEmuNoRoom);
    eq("no room zero", mesh_emu_region_at(0, 0), kMeshEmuNoRoom);
    eq("index offset", mesh_emu_index_at(256, 4), 256 + 64 + 4 * 16);
    if (failures) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("PASS: region size, 256-byte alignment, wrap and no-room\n");
    return 0;
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxx", default="clang++")
    a = p.parse_args()

    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mesh-layout-", dir=work) as temp:
        folder = Path(temp).resolve()
        src = folder / "fixture.cpp"
        exe = folder / "fixture.exe"
        src.write_text(FIXTURE, encoding="utf-8")
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-I" + str(ROOT / "gpu" / "vk"), str(src), "-o", str(exe)],
                       check=True)
        # (a MinGW toolchain's runtime DLLs live next to its compiler: put them on the child's PATH)
        import os
        env = dict(os.environ)
        env["PATH"] = str(Path(a.cxx).resolve().parent) + os.pathsep + env.get("PATH", "")
        r = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
        print(r.stdout.strip())
        if r.returncode != 0:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
