cmake_minimum_required(VERSION 3.5)
project(matmul-avx C)

find_package(OpenBLAS REQUIRED)

set(UTILS benchmark.c cmdLine.c wall_time.c)

add_executable(benchmark-naive ${UTILS} dgemm-naive.c)
target_link_libraries(benchmark-naive PRIVATE ${OpenBLAS_LIBRARIES})
target_include_directories(benchmark-naive PRIVATE ${OpenBLAS_INCLUDE_DIRS})

add_executable(benchmark-blas ${UTILS} dgemm-blas.c)
target_link_libraries(benchmark-blas PRIVATE ${OpenBLAS_LIBRARIES})
target_include_directories(benchmark-blas PRIVATE ${OpenBLAS_INCLUDE_DIRS})

add_executable(benchmark-blocked ${UTILS} dgemm-blocked.c)
target_link_libraries(benchmark-blocked PRIVATE ${OpenBLAS_LIBRARIES})
target_include_directories(benchmark-blocked PRIVATE ${OpenBLAS_INCLUDE_DIRS})
target_compile_options(benchmark-blocked PRIVATE -Ofast -DNDEBUG -Winline -march=native)