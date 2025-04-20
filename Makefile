#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3


#################################################################################
# COMMANDS                                                                      #
#################################################################################

test:
	@echo ${PROJECT_DIR}
	@echo ${PYTHON_INTERPRETER}

requirements:
	pip3 install -r requirements.txt