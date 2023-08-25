#!/usr/bin/env python3
#
# Script to run HeCBench benchmarks and gather results

import re, time, sys, subprocess, multiprocessing, os
import argparse
import json
import csv
from collections import defaultdict

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to extract kernel details from the JSON data
def extract_kernel_details(data):
    kernel_details = defaultdict(list)
    for event in data['traceEvents']:
        if 'name' in event and 'dur' in event and ('Kernel' in event['name'] or 'kernel' in event['name']) and 'args' in event:
            kernel_name = event['name']
            kernel_duration = event['dur']
            if 'detail' in event['args']:
                kernel_detail = event['args']['detail']
                kernel_detail_tuple = tuple(kernel_detail.strip(';').split(';'))
                kernel_details[kernel_name].append((kernel_duration, kernel_detail_tuple))
            else:
                kernel_details[kernel_name].append((kernel_duration,))
    return kernel_details

# Function to calculate statistics for each kernel name
def calculate_kernel_statistics(kernel_details):
    kernel_statistics = {}
    for kernel_name, details_list in kernel_details.items():
        max_duration = max(details_list, key=lambda x: x[0])[0]
        min_duration = min(details_list, key=lambda x: x[0])[0]
        total_duration = sum(d[0] for d in details_list)
        avg_duration = total_duration / len(details_list)
        kernel_statistics[kernel_name] = {
            'max_duration': max_duration,
            'min_duration': min_duration,
            'avg_duration': avg_duration,
            'total_duration': total_duration
        }
    return kernel_statistics

# Function to save kernel statistics to a CSV file
def save_kernel_statistics_to_csv(kernel_statistics, output_file, output_folder):
    output_file_path = os.path.join(output_folder, output_file)
    with open(output_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Kernel Name', 'Max Duration', 'Min Duration', 'Average Duration', 'Total Duration', 'Detail'])
        for kernel_name, details in kernel_statistics.items():
            detail = ''
            if 'detail' in details:
                detail = ';'.join(details['detail'])
            csv_writer.writerow([
                kernel_name,
                details['max_duration'],
                details['min_duration'],
                details['avg_duration'],
                details['total_duration'],
                detail
            ])
    print(f"Kernel statistics saved to {output_file}")

class Benchmark:
    def __init__(self, args, name, invert = False):
        # print(name)
        if name.endswith('sycl'):
            self.MAKE_ARGS = ['GCC_TOOLCHAIN="{}"'.format(args.gcc_toolchain)]
            if args.sycl_type == 'cuda':
                self.MAKE_ARGS.append('CUDA=yes')
                self.MAKE_ARGS.append('CUDA_ARCH=sm_{}'.format(args.nvidia_sm))
            elif args.sycl_type == 'hip':
                self.MAKE_ARGS.append('HIP=yes')
                self.MAKE_ARGS.append('HIP_ARCH={}'.format(args.amd_arch))
            elif args.sycl_type == 'opencl':
                self.MAKE_ARGS.append('CUDA=no')
                self.MAKE_ARGS.append('HIP=no')
        elif name.endswith('cuda'):
            self.MAKE_ARGS = ['CUDA_ARCH=sm_{}'.format(args.nvidia_sm)]
        elif name.endswith('omp'):
            self.MAKE_ARGS = ['-f']
            self.MAKE_ARGS.append('Makefile.nclang')
        else:
            self.MAKE_ARGS = []

        if args.extra_compile_flags:
            flags = args.extra_compile_flags.replace(',',' ')
            self.MAKE_ARGS.append('EXTRA_CFLAGS={}'.format(flags))

        if args.bench_dir:
            self.path = os.path.realpath(os.path.join(args.bench_dir, name))
        else:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', name)

        self.name = name
        # self.args = run_args
        self.invert = invert
        self.clean = args.clean
        self.verbose = args.verbose

    def compile(self):
        if self.clean:
            subprocess.run(["make", "clean"], cwd=self.path).check_returncode()
            time.sleep(1) # required to make sure clean is done before building, despite run waiting on the invoked executable

        out = subprocess.DEVNULL
        if self.verbose:
            out = subprocess.PIPE

        proc = subprocess.run(["make"] + self.MAKE_ARGS, cwd=self.path, stdout=out, stderr=subprocess.STDOUT, encoding="ascii")
        
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f'Failed compilation in {self.path}.\n{e}')
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            raise(e)

        if self.verbose:
            print(proc.stdout)

    def run(self): 
        input_file = self.name + '.json'  
        output_csv_file = self.name + '.csv'  
        my_env = os.environ.copy()
        my_env["LIBOMPTARGET_PROFILE"] = input_file

        out = subprocess.DEVNULL
        if self.verbose:
            out = subprocess.PIPE

        proc = subprocess.run(['make'] + self.MAKE_ARGS + ['run'], cwd=self.path, stdout=out, stderr=subprocess.STDOUT, encoding="ascii", env=my_env)
        
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f'Failed at runtime in {self.path}.\n{e}')
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            raise(e)

        if self.verbose:
            print(proc.stdout)
            
        data = load_json(self.path + '/' + input_file)
        kernel_details = extract_kernel_details(data)
        kernel_statistics = calculate_kernel_statistics(kernel_details)
        if not os.path.exists('kernel-stats'):
            os.makedirs('kernel-stats')
        save_kernel_statistics_to_csv(kernel_statistics, output_csv_file, 'kernel-stats')
        
        return 0


def comp(b):
    print("compiling: {}".format(b.name))
    b.compile()

def main():
    parser = argparse.ArgumentParser(description='HeCBench runner')
    parser.add_argument('--output', '-o',
                        help='Output file for csv results')
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='Repeat benchmark run')
    parser.add_argument('--sycl-type', '-s', choices=['cuda', 'hip', 'opencl'], default='cuda',
                        help='Type of SYCL device to use')
    parser.add_argument('--nvidia-sm', type=int, default=60,
                        help='NVIDIA SM version')
    parser.add_argument('--amd-arch', default='gfx908',
                        help='AMD Architecture')
    parser.add_argument('--gcc-toolchain', default='',
                        help='GCC toolchain location')
    parser.add_argument('--extra-compile-flags', '-e', default='',
                        help='Additional compilation flags (inserted before the predefined CFLAGS)')
    parser.add_argument('--clean', '-c', action='store_true',
                        help='Clean the builds')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Clean the builds')
    parser.add_argument('--runtime', '-R', choices=['srun', 'jsrun'], default='',
                        help='If you need to specify srun or jsrun at runtime')
    parser.add_argument('--bench-dir', '-b',
                        help='Benchmark directory')
    parser.add_argument('--bench-data', '-d',
                        help='Benchmark data')
    parser.add_argument('--bench-fails', '-f',
                        help='List of failing benchmarks to ignore')
    parser.add_argument('bench', nargs='+',
                        help='Either specific benchmark name or sycl, cuda, or hip')
    

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load benchmark data
    if args.bench_data:
        bench_data = args.bench_data
    else:
        bench_data = os.path.join(script_dir, 'benchmarks', 'subset.json') 

    with open(bench_data) as f:
        benchmarks = json.load(f)

    # Load fail file
    if args.bench_fails:
        bench_fails = os.path.abspath(args.bench_fails)
    else:
        bench_fails = os.path.join(script_dir, 'benchmarks', 'subset-fails.txt')

    with open(bench_fails) as f:
        fails = f.read().splitlines()

    # Build benchmark list
    benches = []
    for b in args.bench:
        if b in ['sycl', 'cuda', 'hip', 'omp']:
            benches.extend([Benchmark(args, k, *v)
                            for k, v in benchmarks.items()
                            if k.endswith(b) and k not in fails])
            continue

        benches.append(Benchmark(args, b, *benchmarks[b]))

    t0 = time.time()
    try:
        with multiprocessing.Pool() as p:
            p.map(comp, benches)
    except Exception as e:
        print("Compilation failed, exiting")
        print(e)
        sys.exit(1)

    t_compiled = time.time()

    outfile = sys.stdout

    if args.output:
        outfile = open(args.output, 'w')
        print("Test_Name" + "," + "Compiler" + "," + "Status" + "," + "Time", file=outfile)
        
    if args.runtime:
        runtime_command = args.runtime
    else:
        runtime_command = ''

    for b in benches:
        try:
            if args.verbose:
                print("running: {}".format(b.name))

            t0_runtime = time.time()

            res = []
            for i in range(args.repeat):
                print("starting")
                b.run()
                t_runtime = time.time()
                res.append(str(t_runtime-t0_runtime))
    
            if args.repeat:
                if len(res) > 0:
                    avg_res = sum(map(float, res)) / len(res)
                    print("Average result:", avg_res)
                else:
                    print("No results to average")


            print(b.name + "," + "clang" + "," + "PASS" + ",{},{}".format(avg_res, t_runtime-t0_runtime), file=outfile)

            # print(b.name + "," + "clang" + "," + "PASS" + "," + ",".join(res), file=outfile)
            # print(b.name + "," + ", ".join(res), file=outfile)
        except Exception as err:
            print(b.name + "," + "clang" + "," + "FAIL" + "," + "N/A", file=outfile)
            print("Error running: ", b.name)
            print(err)
    if args.output:
        outfile.close()

    t_done = time.time()
    print("compilation took {} s, runnning took {} s.".format(t_compiled-t0, t_done-t_compiled))

if __name__ == "__main__":
    main()

