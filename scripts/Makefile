###################################################
# HeCBench Makefile for portability and 
# ease of use with different compilers. 
# Only using OpenMP Benchmarks. 
#
# Author: Aaron Jarmusch 
# email: jarmusch@udel.edu
# University of Delaware
###################################################



###################################################
# Source files & Locations
###################################################
override FILE_NCLANG := "createMakefiles-nclang.py"
override FILE_NVC := "createMakefiles-nvc.py"
override FILE_NGCC := "createMakefiles-ngcc.py"
override PWD_HECBENCH := /opt/sollve/hecbench-fork/
override DIR := omp # Variable to hold the directory


###################################################
# OpenMP Versions. This is to support multiple
# versions of the standard in the same testsuite
###################################################
.PHONY: all-nclang
all-nclang: create-nclang auto-run-nclang

.PHONY: all-nvc
all-nvc: create-nvc auto-run-nvc

.PHONY: all-ngcc
all-ngcc: create-ngcc auto-run-ngcc


###################################################
# OpenMP Versions. This is to support multiple
# versions of the standard in the same testsuite
###################################################
.PHONY: run-nvc
run-nvc:
	@clear; \
	cd $(PWD_HECBENCH)$(DIR); \
	make -f Makefile.nvc run; \

# Already exists
# .PHONY: create-nvc
# create-nvc:
# 	python3 $(FILE_NCLANG)

.PHONY: auto-run-nvc
auto-run-nvc:
	@clear; \
	cd scripts; \
	python3 autohecbench-omp-nvc.py $(DIR) --verbose --nvidia-sm 70 -o omp.csv; \
	echo "omp.csv:"; cat omp.csv; \

# .PHONY: clean-nvc
# clean-nvc: 
# 	find /opt/sollve/HeCBench -type f -name "Makefile.nvc" -delete;

###################################################
# OpenMP Versions. This is to support multiple
# versions of the standard in the same testsuite
###################################################
.PHONY: run-nclang
run-nclang:
	@clear; \
	cd $(PWD_HECBENCH)$(DIR); \
	make -f Makefile.nvc run; \

.PHONY: create-nclang
create-nclang:
	python3 $(FILE_NCLANG)

.PHONY: auto-run-nclang
auto-run-nclang:
	@clear
	@cd scripts; \
	python3 autohecbench-omp-nclang.py $(DIR) --verbose -o omp.csv; \
	echo ""; echo "omp.csv:"; cat omp.csv;

.PHONY: clean-nclang
clean-nclang: 
	find $(PWD_HECBENCH) -type f -name "Makefile.nclang" -delete

###################################################
# OpenMP Versions. This is to support multiple
# versions of the standard in the same testsuite
###################################################
.PHONY: run-ngcc
run-ngcc:
	@clear; \
	cd $(PWD_HECBENCH)$(DIR); \
	make -f Makefile.nvc run; \

.PHONY: create-ngcc
create-ngcc:
	python3 $(FILE_NCLANG)

.PHONY: auto-run-ngcc
auto-run-ngcc:
	@clear; \
	cd scripts; \
	python3 autohecbench-omp-nvc.py $(DIR) --verbose --nvidia-sm 70 -o omp.csv; \
	echo "omp.csv:"; cat omp.csv; \

.PHONY: clean-ngcc
clean-ngcc: 
	find /opt/sollve/HeCBench -type f -name "Makefile.ngcc" -delete;

###################################################
# Clean
###################################################

.PHONY: python-clean
python-clean:
	rm -rf kernel-stats
	python3 autohecbench-omp-nclang.py omp -c
	
# edit:
# 	vim ./$(FILE)

# lint:
# 	pylint $(FILE)

# black: 
# 	black $(FILE)

# env: 
# 	export PATH="/usa/jarmusch/.local/bin:$PATH"

# clean:
# 	find /opt/sollve/HeCBench -type f -name "Makefile.nclang" -delete;
