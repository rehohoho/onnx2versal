#ifndef __KERNEL_UTILS_H__
#define __KERNEL_UTILS_H__

#include "aie_api/aie.hpp"

#ifdef EXTERNAL_IO
#define PROFILE_HEADER(stmt)
#define PROFILE_FOOTER 
#else
#define PROFILE_HEADER(stmt) \
  stmt; \
  unsigned cycle_num[2]; \
  aie::tile tile = aie::tile::current(); \
  cycle_num[0] = tile.cycles(); // cycle counter of the AI Engine tile
#define PROFILE_FOOTER \
  cycle_num[1] = tile.cycles(); \
  printf("start = %d,end = %d,total = %d\n", cycle_num[0], cycle_num[1], cycle_num[1] - cycle_num[0]);
#endif

template<typename VECTYPE>
void print_fvec(VECTYPE* vec, int N) {
  for (int i = 0; i < N; i++) 
    printf("%f ", vec[i]);
  printf("\n");
}

template<typename VECTYPE>
void print_vec(VECTYPE* vec, int N) {
  for (int i = 0; i < N; i++) 
    printf("%d ", vec[i]);
  printf("\n");
}

#endif // __KERNEL_UTILS_H__