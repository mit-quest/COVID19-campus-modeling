###########################################################################
# Command Variables
#
# These are usually not overridden by users but can be.
#
PYTHON ?= python3
PIP ?= pip3
UID := $(shell id -u ${USER})

###########################################################################
# Miscellaneous Variables
#
LOCAL_OUTPUTS_LOCATION = local_outputs

###########################################################################
# Virtual Environment Locations
#
# Should not really be changed
#
VENV_LOCATION := venv
VENV_PYTHON := ${VENV_LOCATION}/bin/python
VENV_PIP := ${VENV_LOCATION}/bin/pip

###########################################################################
# Virtual Environment Setup
#
.DEFAULT_GOAL := setup

build_venv:
	@echo ${VENV_LOCATION}
	@mkdir ${VENV_LOCATION}
	${PYTHON} -m venv ${VENV_LOCATION}
	${VENV_PIP} install -q -r models/common/requirements.txt
	venv/bin/pre-commit install

setup: build_venv
	@mkdir ${LOCAL_OUTPUTS_LOCATION}

clean:
	rm -rf ${VENV_LOCATION}
	rm -rf ${LOCAL_OUTPUTS_LOCATION}
