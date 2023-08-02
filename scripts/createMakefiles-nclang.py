import os


def enter_directory(directory_path):
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Change the current working directory to the specified directory
    os.chdir(directory_path)
    print(f"Entered directory '{directory_path}'.")


def edit_file_in_omp_directories(dir_path_root, omp_dirs, file_name, new_file_name):
    for omp_dir in omp_dirs:
        print("######\nEntering Directory: " + omp_dir)

        file_path = os.path.join(dir_path_root, omp_dir, file_name)

        new_file_path = os.path.join(dir_path_root, omp_dir, new_file_name)

        if os.path.isfile(file_path):
            edit_file(file_path, new_file_path)

            print(f"Created file '{new_file_name}' in directory: {omp_dir}")
            print("######\n\n\n")
        else:
            print(f"File '{file_name}' not found in directory: {omp_dir}")
            print("######\n\n\n")


def edit_file(file_path, new_file_path):
    print("Opening Original Makefile for editing....")

    # Open the file, edit its content, and save the changes
    with open(file_path, "r") as file:
        content = file.read()
        content = (
            content.replace("-target x86_64-pc-linux-gnu", "-lm")
            .replace("-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa", "-O3")
            .replace("-Xopenmp-target=amdgcn-amd-amdhsa", "-fopenmp")
            .replace("-march=$(ARCH)", "--offload-arch=native")
        )

    with open(new_file_path, "w") as file:
        file.write(content)


def main():
    os.chdir("..")
    print(os.getcwd())
    dir_path_root = os.getcwd()
    file_name = "Makefile.aomp"
    new_file_name = "Makefile.nclang"

    # enter_directory(dir_path_root)

    print("Getting list of directories....")

    # Find "-omp" directories and edit the file within each directory
    omp_dirs = [
        item
        for item in os.listdir(dir_path_root)
        if os.path.isdir(os.path.join(dir_path_root, item)) and item.endswith("-omp")
    ]

    print("Got List")

    if len(omp_dirs) == 0:
        print("No '-omp' directories found.")
        return

    edit_file_in_omp_directories(dir_path_root, omp_dirs, file_name, new_file_name)


if __name__ == "__main__":
    main()
