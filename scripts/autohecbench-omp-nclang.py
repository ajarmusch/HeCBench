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
    def __init__(self, args, name, res_regex, run_args = [], binary = "main", invert = False):
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
            # print("MAKE_ARGS.append")
            self.MAKE_ARGS = ['-f']
            self.MAKE_ARGS.append('Makefile.nclang')
            self.MAKE_ARGS.append('run')
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
        self.binary = binary
        self.res_regex = res_regex
        self.args = run_args
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

        print(self.path)
        proc = subprocess.run(["make"] + self.MAKE_ARGS , cwd=self.path, stdout=out, stderr=subprocess.STDOUT, encoding="ascii")
        
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
        # cmd = ["./" + self.binary] + self.args
        #cmd = ["srun", "./" + self.binary] + self.args
        # cmd = [runtime_command, "./" + self.binary] + self.args
        my_env = os.environ.copy()
        my_env["LIBOMPTARGET_PROFILE"] = input_file
        cmd = ['./' + self.binary] + self.args
        proc = subprocess.run(cmd, cwd=self.path, stdout=subprocess.PIPE, encoding="ascii", env=my_env)
        out = proc.stdout
        if self.verbose:
            print(" ".join(cmd))
            print(out)
        # print(self.res_regex)
        # pattern = self.res_regex.replace('\\s', '\(s\)')#.replace('\\us', '\(us\)').replace('\\ms', '\(ms\)')
        # pattern = r'Device offloading time = ([0-9.+-e]+)\(s\)'
        #             Device offloading time = ([0-9.+-e]+)\\s"
        # pattern = r'Average execution time of accuracy kernel: ([0-9.+-e]+) \(us\)'
        #             Average execution time of accuracy kernel: ([0-9.+-e]+)\\us",
                    # Average kernel execution time \(w/ shared memory\): ([0-9.]+) \(us\)
        res = re.findall(self.res_regex, out)
        # res = re.findall(pattern, out)
        if not res:
            raise Exception(self.path + ":\nno regex match for " + self.res_regex + " in\n" + out)
        res = sum([float(i) for i in res]) #in case of multiple outputs sum them
        if self.invert:
            res = 1/res
            
        data = load_json(self.path + '/' + input_file)
        kernel_details = extract_kernel_details(data)
        kernel_statistics = calculate_kernel_statistics(kernel_details)
        if not os.path.exists('kernel-stats'):
            os.makedirs('kernel-stats')
        save_kernel_statistics_to_csv(kernel_statistics, output_csv_file, 'kernel-stats')
        
        return res


def comp(b):
    print("compiling: {}".format(b.name))
    b.compile()

def output_gen(compiler):
    with open('./benchmarks/subset-omp-test.json', 'r') as bench_info_file:
        bench_info = json.load(bench_info_file)
        bench_names = list(bench_info.keys())
        print(bench_names[0])

    with open("omp.csv", 'r') as bench_output:
        bench_output_res = list(csv.reader(bench_output))
        print(bench_output_res[0][0])
        #for bench_passes in range(len(bench_output_res)):
            #print(bench_output_res[bench_passes][0])

    with open('output.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Test_Name", "Compiler", "Status", "Time"])
        for a in range(len(bench_names)):
            check_pres=0
            for bench_passes in range(len(bench_output_res)):
                print(bench_names[a])
                if bench_names[a] == (bench_output_res[bench_passes][0]):
                    check_pres = 1
                    pass_val=bench_passes

            if check_pres == 1:
                writer.writerow([bench_output_res[pass_val][0], compiler, "PASS", bench_output_res[pass_val][1]])
            else:
                writer.writerow([bench_names[a], compiler, "FAIL", "N/A"])

def main():
    parser = argparse.ArgumentParser(description='HeCBench runner')
    parser.add_argument('--output', '-o',
                        help='Output file for csv results')
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='Repeat benchmark run')
    parser.add_argument('--warmup', '-w', type=bool, default=True,
                        help='Run a warmup iteration')
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
        bench_data = os.path.join(script_dir, 'benchmarks', 'subset-omp-test.json') 

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
    # print(outfile)
    # print(args.output)
    if args.output:
        outfile = open(args.output, 'w')
        
    if args.runtime:
        runtime_command = args.runtime
    else:
        runtime_command = ''

    for b in benches:
        try:
            if args.verbose:
                print("running: {}".format(b.name))
            
            if args.warmup:
                b.run()

            res = []
            for i in range(args.repeat):
                res.append(str(b.run()))
                
            print(b.name + "," + ", ".join(res), file=outfile)
        except Exception as err:
            print("Error running: ", b.name)
            print(err)
    if args.output:
        outfile.close()

    t_done = time.time()
    print("compilation took {} s, runnning took {} s.".format(t_compiled-t0, t_done-t_compiled))
    output_gen("clang")

if __name__ == "__main__":
    main()

