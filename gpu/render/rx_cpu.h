// The renderer's own CPU timing: a cycle counter and the spin-loop hint (rx_core.cpp, rx_draw.cpp).
//
// x86 uses the TSC and PAUSE. ARM64 (Windows on Snapdragon, Linux on aarch64) has no __rdtsc or _mm_pause: clang's
// __builtin_readcyclecounter() reads Arm's generic counter (cntvct_el0), whose only use here is a per-frame duration,
// and YIELD is the wait hint in a spin loop.
#pragma once
#include <cstdint>

#if defined(_M_ARM64) || defined(__aarch64__)
#  if defined(__clang__) || defined(__GNUC__)
#    define RX_RDTSC() ((uint64_t)__builtin_readcyclecounter())
#    define RX_CPU_RELAX() __asm__ __volatile__("yield")
#  else
#    include <intrin.h>
#    define RX_RDTSC() ((uint64_t)__rdpmccntr64())
#    define RX_CPU_RELAX() __yield()
#  endif
#else
#  include <intrin.h>
#  define RX_RDTSC() ((uint64_t)__rdtsc())
#  define RX_CPU_RELAX() _mm_pause()
#endif
