set(EXAMPLES_CPP
    opencv_test
)

foreach(EXAMPLE ${EXAMPLES_CPP})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} signals)
endforeach()

set(EXAMPLES_CUDA
    
)

foreach(EXAMPLE ${EXAMPLES_CUDA})
    add_executable(${EXAMPLE} ${EXAMPLE}.cu)
    target_link_libraries(${EXAMPLE} signals)
endforeach()