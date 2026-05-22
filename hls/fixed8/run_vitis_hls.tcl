# Vitis HLS — fixed8 (ap_fixed<8,4>)
#   vitis_hls -f run_vitis_hls.tcl

set proj_dir [file normalize [file dirname [info script]]]
cd $proj_dir

exec bash setup_links.sh

set proj_name mnist_fixed8
open_project -reset $proj_name
set_top mnist_cnn

add_files top_kernel.cpp -cflags "-I. -DUSE_VITIS_HLS"
add_files cnn_forward.cpp -cflags "-I. -DUSE_VITIS_HLS"
add_files -tb tb_csim.cpp -cflags "-I. -DUSE_VITIS_HLS -DCSIM_LOCAL"

open_solution -reset sol1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

csim_design
csynth_design

# Synth-only (skip csim): vitis_hls -f run_csynth_only.tcl

exit
