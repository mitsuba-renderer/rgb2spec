// Macros to help compilers autovectorize code in this project

#pragma once

#if defined(__clang__)
#  define RGB2SPEC_FP_FAST
#  define RGB2SPEC_FP_REASSOC _Pragma("clang fp reassociate(on) contract(fast)")
#elif defined(__GNUC__)
#  define RGB2SPEC_FP_FAST __attribute__((optimize("-fassociative-math", \
       "-fno-signed-zeros", "-fno-trapping-math", "-fno-math-errno", "-ffp-contract=fast")))
#  define RGB2SPEC_FP_REASSOC
#else
#  define RGB2SPEC_FP_FAST
#  define RGB2SPEC_FP_REASSOC
#endif

#if defined(_MSC_VER)
#  define RGB2SPEC_FP_PUSH _Pragma("float_control(precise, off, push)")
#  define RGB2SPEC_FP_POP  _Pragma("float_control(pop)")
#else
#  define RGB2SPEC_FP_PUSH
#  define RGB2SPEC_FP_POP
#endif
