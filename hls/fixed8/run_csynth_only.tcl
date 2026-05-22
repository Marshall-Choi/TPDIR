# Vitis HLS — csynth only (skip csim, lighter compile flags)
#   vitis_hls -f run_csynth_only.tcl
#
# Use after Mac/Linux csim passes. If synth still hangs 2+ hours, see SYNTH_NOTES.txt

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

# 20 ns = easier timing closure on first bring-up (optimize later)
create_clock -period 20 -name default

# Reduce scheduler blow-up on deep loop nests
if {[catch {config_compile -pipeline_loops 32}]} {}
if {[catch {config_compile -disable_unroll_loops}]} {}

csynth_design

exit
