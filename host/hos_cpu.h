/* Yield the core inside a spin loop: x86's PAUSE, ARM64's YIELD.

   ARM64 Windows has no _mm_pause, and clang's <emmintrin.h> refuses to be included for an aarch64 target at all, so
   every spin site in host/ asks this header instead. */
#ifndef HOS_CPU_H
#define HOS_CPU_H

#if defined(_M_ARM64) || defined(__aarch64__)
#  if defined(_MSC_VER)
#    include <intrin.h>
#    define HOS_CPU_RELAX() __yield()
#  else
#    define HOS_CPU_RELAX() __asm__ __volatile__("yield")
#  endif
#elif defined(_MSC_VER)
#  include <intrin.h>
#  define HOS_CPU_RELAX() _mm_pause()
#else
#  include <xmmintrin.h>
#  define HOS_CPU_RELAX() _mm_pause()
#endif

#endif
