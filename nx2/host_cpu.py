"""The CPU architecture and x86-64 level this PC runs (nx2/build.py's --target and -march; issues 363, 640).

The game is built on the PC it runs on. x86-64 machines get the highest level their CPU has (x86-64-v3, -v2, or the
x86-64 baseline). ARM64 machines (Windows on Snapdragon, Linux on aarch64) are built natively for their own
architecture, with the ARMv8.2 extensions the Switch 2's guest uses, instead of being run under emulation.

RX_MARCH, read by build.py, still chooses the -march value. HOOKSHOT_ARCH=x64|arm64 overrides the architecture (a cross
build, or a test).
"""
import os

# The Oryon cores (Snapdragon X Elite) are ARMv8.2-A with FP16, dot-product, NEON and crypto: what the guest needs in
# the lifted code's own arithmetic. -mtune=native (build.py) then tunes for the core that builds it.
ARM64_MARCH = "armv8.2-a+fp16+dotprod+simd+crypto"


def host_arch():
    """'arm64' or 'x64': the architecture this PC runs. HOOKSHOT_ARCH overrides it."""
    want = os.environ.get("HOOKSHOT_ARCH", "").lower()
    if want in ("arm64", "aarch64"):
        return "arm64"
    if want in ("x64", "x86_64", "amd64"):
        return "x64"
    if os.name == "nt":
        # (PROCESSOR_ARCHITECTURE is AMD64 on x64 Windows and ARM64 on Windows on ARM; ARM64EC and ARM64X also report
        # ARM64, so both are treated as ARM64 here)
        return "arm64" if os.environ.get("PROCESSOR_ARCHITECTURE", "").upper() in ("ARM64", "ARM64EC") else "x64"
    try:
        import platform
        machine = platform.machine().lower()
    except Exception:
        machine = ""
    return "arm64" if machine in ("arm64", "aarch64") else "x64"


def host_march():
    """The -march value for this PC: an ARMv8.2-A extension set on ARM64, an x86-64 level otherwise."""
    if host_arch() == "arm64":
        return ARM64_MARCH
    want = ("avx2", "fma", "bmi2", "movbe")
    try:
        if os.name == "nt":
            import ctypes
            pf = ctypes.windll.kernel32.IsProcessorFeaturePresent
            # (PF_AVX2_INSTRUCTIONS_AVAILABLE 40; PF_SSE4_2_INSTRUCTIONS_AVAILABLE 38. Windows reports no FMA/BMI2 bit, but
            # every AVX2 CPU has them, as x86-64-v3 counts them together)
            if pf(40):
                return "x86-64-v3"
            return "x86-64-v2" if pf(38) else "x86-64"
        flags = open("/proc/cpuinfo").read().split("flags", 1)[1].split(chr(10), 1)[0].split()
        if all(f in flags for f in want):
            return "x86-64-v3"
        return "x86-64-v2" if all(f in flags for f in ("sse4_2", "popcnt", "ssse3")) else "x86-64"
    except Exception:
        return "x86-64-v3"
