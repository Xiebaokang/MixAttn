function(locate_file_dir FILENAME OUTVAR)
  # recursively look for the file in the parent directories, until the root directory is reached
  # the function sets a variable with name "FOUND_FILE_DIR" to the directory where the file is found in the parent scope 
  set(current_dir ${CMAKE_CURRENT_LIST_DIR})
  while(NOT EXISTS "${current_dir}/${FILENAME}")
    get_filename_component(current_dir ${current_dir} DIRECTORY)
    if("${current_dir}" STREQUAL "/")
      message(FATAL_ERROR "File ${FILENAME} not found in the parent directories")
    endif()
  endwhile()
  set(${OUTVAR} ${current_dir} PARENT_SCOPE)
endfunction()

function(assert_dir_exists DIR)
  if(NOT EXISTS ${DIR})
    message(FATAL_ERROR "Directory ${DIR} does not exist")
  endif()
endfunction()
