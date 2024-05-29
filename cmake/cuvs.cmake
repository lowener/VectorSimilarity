if(USE_CUDA)
	# Set which version of RAPIDS to use
	set(RAPIDS_VERSION 24.06)
	# Set which version of CUVS to use (defined separately for testing
	# minimal dependency changes if necessary)
	set(CUVS_VERSION "${RAPIDS_VERSION}")
	set(CUVS_FORK "rapidsai")
	set(CUVS_PINNED_TAG "branch-${RAPIDS_VERSION}")

	# Download CMake file for bootstrapping RAPIDS-CMake, a utility that
	# simplifies handling of complex RAPIDS dependencies
	if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
		file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION}/RAPIDS.cmake
			${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
	endif()
	include(${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)

	# General tool for orchestrating RAPIDS dependencies
	include(rapids-cmake)
	# CPM helper functions with dependency tracking
	include(rapids-cpm)
	rapids_cpm_init()
	# Common CMake CUDA logic
	include(rapids-cuda)
	# Include required dependencies in Project-Config.cmake modules
	# include(rapids-export)  TODO(wphicks)
	# Functions to find system dependencies with dependency tracking
	include(rapids-find)

	# Correctly handle supported CUDA architectures
	#    (From rapids-cuda)
	rapids_cuda_init_architectures(VectorSimilarity)

	# Find system CUDA toolkit
	rapids_find_package(CUDAToolkit REQUIRED)
	
	function(find_and_configure_cuvs)
		set(oneValueArgs VERSION FORK PINNED_TAG BUILD_SHARED_LIBS)
		cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
			"${multiValueArgs}" ${ARGN} )
	
		set(CUVS_COMPONENTS "")
		if(PKG_BUILD_SHARED_LIBS)
			string(APPEND CUVS_COMPONENTS " compiled")
		endif()
		# Invoke CPM find_package()
		#     (From rapids-cpm)
		rapids_cpm_find(cuvs ${PKG_VERSION}
			GLOBAL_TARGETS      cuvs::cuvs
			BUILD_EXPORT_SET    VectorSimilarity-exports
			INSTALL_EXPORT_SET  VectorSimilarity-exports
			COMPONENTS          ${CUVS_COMPONENTS}
			CPM_ARGS
			GIT_REPOSITORY https://github.com/${PKG_FORK}/cuvs.git
			GIT_TAG        ${PKG_PINNED_TAG}
			SOURCE_SUBDIR  cpp
			OPTIONS
			"BUILD_TESTS OFF"
			"BUILD_SHARED_LIBS ${PKG_BUILD_SHARED_LIBS}"
		)
		if(cuvs_ADDED)
			message(VERBOSE "VectorSimilarity: Using cuVS located in ${cuvs_SOURCE_DIR}")
		else()
			message(VERBOSE "VectorSimilarity: Using cuVS located in ${cuvs_DIR}")
		endif()
	endfunction()
	
	# Change pinned tag here to test a commit in CI
	# To use a different CUVS locally, set the CMake variable
	# CPM_cuvs_SOURCE=/path/to/local/cuvs
	find_and_configure_cuvs(VERSION    ${CUVS_VERSION}.00
		FORK               ${CUVS_FORK}
		PINNED_TAG         ${CUVS_PINNED_TAG}
		BUILD_SHARED_LIBS  OFF
	)
endif()
