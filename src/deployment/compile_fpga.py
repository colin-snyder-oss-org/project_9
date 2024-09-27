# src/deployment/compile_fpga.py
import os
import subprocess

def compile_model_for_fpga(vivado_path, project_dir):
    os.environ['XILINX_VIVADO'] = vivado_path
    make_command = f"make -C {project_dir}"
    subprocess.run(make_command, shell=True, check=True)
    print("FPGA bitstream generated successfully.")
