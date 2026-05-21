# Vitis HLS — MNIST BNN
#   vitis_hls -f run_vitis_hls.tcl

set proj_dir [file normalize [file dirname [info script]]]
cd $proj_dir

exec bash setup_links.sh

set proj_name mnist_bnn
open_project -reset $proj_name
set_top mnist_bnn

add_files top_kernel.cpp -cflags "-I."
add_files cnn_forward.cpp -cflags "-I."
add_files cnn_forward.h
add_files cnn_dims.h
add_files -tb tb_csim.cpp -cflags "-I. -DCSIM_LOCAL"

open_solution -reset sol1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

csim_design
csynth_design
# cosim_design
# export_design -format ip_catalog -output ../export_ip_bnn

exit
