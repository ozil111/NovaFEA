set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_ENV_PASSTHROUGH PATH)

set(VCPKG_CMAKE_SYSTEM_NAME MinGW)
set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)

# libstdc++ headers may reference winpthread (e.g. __gthread_*); lld does not link it implicitly like g++.
string(APPEND VCPKG_LINKER_FLAGS " -lwinpthread")

# Chainloads vcpkg-mingw-clang-toolchain.cmake (Clang + MinGW target, libstdc++ from MinGW)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/vcpkg-mingw-clang-toolchain.cmake")
