#! /usr/bin/bash

package_name="kernel_extraction"
test_script="test_g_method_second_cpu.py"
time_script="time_g_method_second_cpu.py"

# find out if building debug or release
if ! ([ "$1" = "debug" ] || [ "$1" = "release" ]); then 
	build_mode="debug"
else
	build_mode=$1
fi
echo "Running in: $build_mode mode"

# if build succesfull, link .so to projects top library 
# for debug, exec test script
# for release exec the timing script
if [ "$build_mode" = "release" ]; then
	RUSTFLAGS="-C opt-level=3 -C debuginfo=0 -C target-cpu=native" cargo build --release
else
	cargo build
fi
if [ $? -eq 0 ]; then
	ln -sf target/${build_mode}/lib${package_name}.so ${package_name}.so
	if [ "$build_mode" = "debug" ]; then
		python "test/$test_script"
	else
		python "test/$time_script"
	fi
fi
