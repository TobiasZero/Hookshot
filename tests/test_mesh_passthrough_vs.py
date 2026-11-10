"""The mesh-emulation pass-through vertex shader is valid SPIR-V (Phase 3, gpu/vk/mesh_passthrough_vs.cpp).

It is generated with Sirit, so this test builds the generator and the shader library on the host, runs it, and reads the
module back: no GPU, no game files, no shader compiler needed. With --spirv-val the module is validated too.

python -I -B tests/test_mesh_passthrough_vs.py --headers <SPIRV-Headers/include> [--cxx clang++] [--spirv-val spirv-val]
"""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include "mesh_passthrough_vs.h"
#include <cstdio>
#include <vector>

static bool write_words(const char* path, const std::vector<uint32_t>& words) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const size_t written = std::fwrite(words.data(), sizeof(uint32_t), words.size(), f);
    std::fclose(f);
    return written == words.size();
}

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    if (!write_words(argv[1], mkwvk::mesh_passthrough_vs())) return 2;
    if (!write_words(argv[2], mkwvk::mesh_finalize_spv(1, 4, 100, 200, 3))) return 3;
    return 0;
}
'''


def instructions(words):
    """(opcode, operands) of every instruction in a SPIR-V module."""
    i = 5   # (magic, version, generator, bound, schema)
    while i < len(words):
        word = words[i]
        count = word >> 16
        opcode = word & 0xFFFF
        if count == 0 or i + count > len(words):
            raise AssertionError("malformed SPIR-V instruction at word %d" % i)
        yield opcode, words[i + 1:i + count]
        i += count


def spv_string(operands):
    raw = b"".join(struct.pack("<I", w) for w in operands)
    return raw.split(b"\0", 1)[0].decode("ascii", "replace")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headers", type=Path, required=True, help="SPIRV-Headers include dir (spirv/unified1)")
    p.add_argument("--cxx", default="clang++")
    p.add_argument("--spirv-val", default=None)
    a = p.parse_args()

    sirit = ROOT / "gpu" / "vk" / "sirit"
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mesh-vs-", dir=work) as temp:
        folder = Path(temp).resolve()
        fixture = folder / "fixture.cpp"
        exe = folder / "fixture.exe"
        out = folder / "passthrough.spv"
        fin = folder / "finalize.spv"
        fixture.write_text(FIXTURE, encoding="utf-8")
        sources = sorted((sirit / "src").rglob("*.cpp"))
        subprocess.run([a.cxx, "-std=c++20", "-O1", "-I" + str(ROOT / "gpu" / "vk"),
                        "-I" + str(sirit / "include"), "-I" + str(sirit / "src"), "-I" + str(a.headers),
                        str(fixture), str(ROOT / "gpu" / "vk" / "mesh_passthrough_vs.cpp"), *map(str, sources),
                        "-o", str(exe)], check=True)
        # (a MinGW toolchain's C++ runtime DLLs live next to its compiler: put them on the child's PATH)
        env = dict(os.environ)
        env["PATH"] = str(Path(a.cxx).resolve().parent) + os.pathsep + env.get("PATH", "")
        subprocess.run([str(exe), str(out), str(fin)], check=True, env=env)
        words = list(struct.unpack("<%dI" % (out.stat().st_size // 4), out.read_bytes()))
        fin_words = list(struct.unpack("<%dI" % (fin.stat().st_size // 4), fin.read_bytes()))
        # (the files live only as long as this folder: validate here, not after it is gone)
        if a.spirv_val:
            subprocess.run([a.spirv_val, "--target-env", "vulkan1.3", str(out)], check=True)
            subprocess.run([a.spirv_val, "--target-env", "vulkan1.3", str(fin)], check=True)

    assert words[0] == 0x07230203, "not a SPIR-V module: %#x" % words[0]
    assert words[1] >> 16 == 1 and (words[1] >> 8) & 0xFF == 3, "expected SPIR-V 1.3, got %#x" % words[1]
    assert words[3] > 1, "module bound %d" % words[3]

    entry_points = [(ops[0], spv_string(ops[2:])) for op, ops in instructions(words) if op == 15]
    assert entry_points == [(0, "main")], "expected one Vertex entry point 'main', got %r" % (entry_points,)

    # (the record read is an OpAccessChain into a StorageBuffer; the store writes the Position builtin)
    opcodes = {op for op, _ in instructions(words)}
    assert 65 in opcodes, "no OpAccessChain"
    assert 61 in opcodes, "no OpLoad"
    assert 62 in opcodes, "no OpStore"

    # ---- the finalize pass: one GLCompute invocation turning the index counter into the indirect command ----
    assert fin_words[0] == 0x07230203, "finalize: not a SPIR-V module"
    fin_entry = [(ops[0], spv_string(ops[2:])) for op, ops in instructions(fin_words) if op == 15]
    assert fin_entry == [(5, "main")], "finalize: expected one GLCompute entry point, got %r" % (fin_entry,)
    fin_opcodes = {op for op, _ in instructions(fin_words)}
    assert 16 in fin_opcodes, "finalize: no OpExecutionMode (LocalSize)"
    assert 62 in fin_opcodes, "finalize: no OpStore"

    if a.spirv_val:
        pass   # (validated above, while the module files existed)
    print("PASS: %d-word pass-through VS (Vertex, storage load -> gl_Position) and %d-word finalize pass "
          "(GLCompute, counter -> indirect command)" % (len(words), len(fin_words)))


if __name__ == "__main__":
    main()
