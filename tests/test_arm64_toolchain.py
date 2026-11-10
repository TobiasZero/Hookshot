"""The ARM64 build target (nx2/host_cpu.py, third_party/a64-kit/tools/toolchain.py, the port guards in host/).

Game-free: no compiler or dump is needed, only the target strings and the source guards that keep the x64 paths.
The Windows-on-ARM build is native (aarch64-w64-mingw32 / aarch64-pc-windows-msvc), not an x64 build run under
emulation; -mcx16 is x86-only, so it must not reach an aarch64 command line.

python -I -B tests/test_arm64_toolchain.py
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host_cpu = load("host_cpu_arm64_test", ROOT / "nx2" / "host_cpu.py")
toolchain = load("toolchain_arm64_test", ROOT / "third_party" / "a64-kit" / "tools" / "toolchain.py")


class Arch(unittest.TestCase):
    def test_arm64_windows_is_native(self):
        with mock.patch.object(host_cpu.os, "name", "nt"), \
                mock.patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "ARM64", "HOOKSHOT_ARCH": ""}):
            self.assertEqual(host_cpu.host_arch(), "arm64")
            self.assertEqual(toolchain.arch(), "arm64")

    def test_override_wins(self):
        with mock.patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "ARM64", "HOOKSHOT_ARCH": "x64"}):
            self.assertEqual(host_cpu.host_arch(), "x64")
            self.assertEqual(toolchain.arch(), "x64")


class Triples(unittest.TestCase):
    def target(self, arch, mingw="C:/llvm-mingw", windows=True):
        env = {"HOOKSHOT_ARCH": arch, "PROCESSOR_ARCHITECTURE": "AMD64"}
        if mingw:
            env["HOOKSHOT_MINGW"] = mingw
        with mock.patch.dict(os.environ, env), mock.patch.object(toolchain.os, "name", "nt" if windows else "posix"), \
                mock.patch.object(toolchain, "windows", return_value=windows):
            return toolchain.native_target()

    def test_mingw_triples(self):
        self.assertIn("--target=aarch64-w64-mingw32", self.target("arm64"))
        self.assertIn("--target=x86_64-w64-mingw32", self.target("x64"))

    def test_msvc_triples(self):
        self.assertIn("--target=aarch64-pc-windows-msvc", self.target("arm64", mingw=""))
        self.assertIn("--target=x86_64-pc-windows-msvc", self.target("x64", mingw=""))


class PeMachine(unittest.TestCase):
    """A DLL of the other architecture cannot load into the game: the build reads the PE header, not the file name."""

    def write_pe(self, d, machine):
        head = bytearray(0x40)
        head[:2] = b"MZ"
        head[0x3C:0x40] = (0x40).to_bytes(4, "little")
        body = b"PE\0\0" + machine.to_bytes(2, "little") + b"\0\0"
        path = os.path.join(d, "t.dll")
        with open(path, "wb") as f:
            f.write(head + body)
        return path

    def test_machines(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(toolchain.pe_machine(self.write_pe(d, 0xAA64)), 0xAA64)
            self.assertEqual(toolchain.pe_machine(self.write_pe(d, 0x8664)), 0x8664)
            bogus = os.path.join(d, "bogus.dll")
            with open(bogus, "wb") as f:
                f.write(b"not a PE file")
            self.assertEqual(toolchain.pe_machine(bogus), 0)

    def test_wanted_machine_follows_the_target(self):
        with mock.patch.dict(os.environ, {"HOOKSHOT_ARCH": "arm64"}):
            self.assertEqual(toolchain.wanted_machine(), 0xAA64)
        with mock.patch.dict(os.environ, {"HOOKSHOT_ARCH": "x64"}):
            self.assertEqual(toolchain.wanted_machine(), 0x8664)


class PortGuards(unittest.TestCase):
    """The architecture-specific code stays guarded, so x64 builds keep their own paths (EXECUTION RULE 4)."""

    def test_context_is_arch_aware(self):
        glue = (ROOT / "host" / "mkw_glue.c").read_text(encoding="utf-8")
        self.assertIn("#if defined(_M_ARM64) || defined(__aarch64__)", glue)
        self.assertIn("#define HOS_CTX_PC(c) ((c).Pc)", glue)
        self.assertIn("#define HOS_CTX_PC(c) ((c).Rip)", glue)
        # (ARM64 has no Dr0-Dr7: the breakpoint watch is compiled out rather than left to fail the build)
        self.assertIn("#if !HOS_ARM64", glue)
        self.assertNotIn("ctx.Rip", glue)   # (only the macro's (c).Rip remains)

    def test_build_flags_split_by_arch(self):
        build = (ROOT / "nx2" / "build.py").read_text(encoding="utf-8")
        self.assertIn('if ARCH == "arm64":', build)
        self.assertIn('"-mtune=native"', build)   # (native builds only: not across a cross build)
        self.assertIn('CPU = ["-march=" + MARCH, "-mcx16", "-ffp-contract=off"]', build)
        # (-mcx16 must not survive in a flag list that aarch64 also compiles)
        self.assertNotIn('"-fno-strict-aliasing", "-mcx16"', build)

    def test_spin_loops_use_the_portable_relax(self):
        for name in ("hos_svc.c", "hos_thread.c", "nn_os_sync.c"):
            text = (ROOT / "host" / name).read_text(encoding="utf-8")
            self.assertIn("HOS_CPU_RELAX()", text, name)
            self.assertNotIn("_mm_pause()", text, name)

    def test_vendored_a64_runtime_has_an_aarch64_fpu_path(self):
        # (a64-kit's FPX1 fast path is the host FPU on ARM64: RECOMP_FPX_HOST 1, no x86 intrinsics involved)
        text = (ROOT / "third_party" / "a64-kit" / "lift" / "suyu" / "arm64_to_c.h").read_text(encoding="utf-8")
        self.assertIn("defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))", text)


if __name__ == "__main__":
    unittest.main()
