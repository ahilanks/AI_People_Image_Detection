include("${CONFIG_FILE}")
set(prefix "COPYFILES: ")

set(use_symlink 0)
if(IS_SYMLINK "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/symlink_test")
  set(use_symlink 1)
endif()

set(update_dephelper 0)
if(DEFINED DEPHELPER AND NOT EXISTS "${DEPHELPER}")
  set(update_dephelper 1)
endif()

set(__state "")

macro(copy_file_ src dst prefix)
  string(REPLACE "${CMAKE_BINARY_DIR}/" "" dst_name "${dst}")
  set(local_update 0)
  if(NOT EXISTS "${dst}")
    set(local_update 1)
  endif()
  if(use_symlink)
    if(local_update OR NOT IS_SYMLINK "${dst}")
      #message("${prefix}Symlink: '${dst_name}' ...")
    endif()
    get_filename_component(target_path "${dst}" PATH)
    file(MAKE_DIRECTORY "${target_path}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${src}" "${dst}"
        RESULT_VARIABLE SYMLINK_RESULT)
    if(NOT SYMLINK_RESULT EQUAL 0)
      #message("Symlink failed, fallback to 'copy'")
      set(use_symlink 0)
    endif()
  endif()
  if(NOT use_symlink)
    if("${src}" IS_NEWER_THAN "${dst}" OR IS_SYMLINK "${dst}")
      file(REMOVE "${dst}") # configure_file(COPYONLY) doesn't update timestamp sometimes
      set(local_update 1)
    endif()
    if(local_update)
      #message("${prefix}Copying: '${dst_name}' ...")
      configure_file(${src} ${dst} COPYONLY)
    else()
      #message("${prefix}Up-to-date: '${dst_name}'")
    endif()
  endif()
  if(local_update)
    set(update_dephelper 1)
  endif()
  file(TIMESTAMP "${dst}" dst_t UTC)
  set(__state "${__state}${dst_t} ${dst}\n")
endmacro()

if(NOT DEFINED COPYLIST_VAR)
  set(COPYLIST_VAR "COPYLIST")
endif()
list(LENGTH ${COPYLIST_VAR} __length)
#message("${prefix}... ${__length} entries (${COPYLIST_VAR})")
foreach(id ${${COPYLIST_VAR}})
  set(src "${${COPYLIST_VAR}_SRC_${id}}")
  set(dst "${${COPYLIST_VAR}_DST_${id}}")
  if(NOT EXISTS ${src})
    message(FATAL_ERROR "Source file/dir is missing: ${src} (${CONFIG_FILE})")
  endif()
  set(_mode "COPYFILE")
  if(DEFINED ${COPYLIST_VAR}_MODE_${id})
    set(_mode "${${COPYLIST_VAR}_MODE_${id}}")
  endif()
  if(_mode STREQUAL "COPYFILE")
    #message("... COPY ${src} => ${dst}")
    copy_file_("${src}" "${dst}" "${prefix}    ")
  elseif(_mode STREQUAL "COPYDIR")
    get_filename_component(src_name "${src}" NAME)
    get_filename_component(src_path "${src}" PATH)
    get_filename_component(src_name2 "${src_path}" NAME)

    set(src_glob "${src}/*")
    if(DEFINED ${COPYLIST_VAR}_GLOB_${id})
      set(src_glob "${${COPYLIST_VAR}_GLOB_${id}}")
    endif()
    file(GLOB_RECURSE _files RELATIVE "${src}" ${src_glob})
    list(LENGTH _files __length)
    #message("${prefix}    ... directory '.../${src_name2}/${src_name}' with ${__length} files")
    foreach(f ${_files})
      if(NOT EXISTS "${src}/${f}")
        message(FATAL_ERROR "COPY ERROR: Source file is missing: ${src}/${f}")
      endif()
      copy_file_("${src}/${f}" "${dst}/${f}" "${prefix}        ")
    endforeach()
  endif()
endforeach()

set(STATE_FILE "${CONFIG_FILE}.state")
if(EXISTS "${STATE_FILE}")
  file(READ "${STATE_FILE}" __prev_state)
else()
  set(__prev_state "")
endif()
if(NOT "${__state}" STREQUAL "${__prev_state}")
  file(WRITE "${STATE_FILE}" "${__state}")
  #message("${prefix}Updated!")
  set(update_dephelper 1)
endif()

if(NOT update_dephelper)
  #message("${prefix}All files are up-to-date.")
elseif(DEFINED DEPHELPER)
  file(WRITE "${DEPHELPER}" "")
endif()
