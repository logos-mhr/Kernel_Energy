cmake_minimum_required(VERSION 2.6)
project(cuda_pcg)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA_VERSION     : ${CUDA_VERSION_STRING}")
  include_directories(${CUDA_INCLUDE_DIRS})
endif (CUDA_FOUND)

include_directories(Cuda)

file(GLOB cuda_pcg_header Cuda/*.cuh)
file(GLOB cuda_pcg_source Cuda/*.cu)

CUDA_ADD_EXECUTABLE(cuda_pcg ${cuda_pcg_header} ${cuda_pcg_source})
target_link_libraries(cuda_pcg
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_cublas_LIBRARY}
  )
