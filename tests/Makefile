# Makefile for testing ShowerPy


#===========================================
# Rules for testing
#===========================================

#------------------------------------
# Main Functions
#------------------------------------

# - - - - - - - - - - - - - - -
# Rules:
# - - - - - - - - - - - - - - -
# Possible make targets (to be make with ```make [xxx]```)
.PHONY : test_all \
	test_qcd_utils test_vector_utils test_parton test_shower_algorithm test_gaugetheoryshower \
	venv update_local

# - - - - - - - - - - - - - - -
# Default
# - - - - - - - - - - - - - - -
# Go through full pipeline to make plots by default
.DEFAULT_GOAL := test_all


venv:
	@echo "Creating virtual environment"
	@python3 -m venv venv
	@echo "Installing ShowerPy"
	@. venv/bin/activate; pip3 install ..
	@. venv/bin/activate; pip3 install pytest

update_local:
	@. venv/bin/activate; pip3 install ..


# Build targets for all test files
test_qcd_utils:
	@echo "Testing QCD utils"
	@. venv/bin/activate; pytest test_qcd_utils.py

test_gaugetheoryshower:
	@echo "Testing GaugeTheoryShower"
	@. venv/bin/activate; pytest test_gaugetheoryshower.py

test_parton:
	@echo "Testing Parton"
	@. venv/bin/activate; pytest test_parton.py

test_shower_algorithm:
	@echo "Testing ShowerAlgorithm"
	@. venv/bin/activate; pytest test_shower_algorithm.py

test_vector_utils:
	@echo "Testing VectorUtils"
	@. venv/bin/activate; pytest test_vector_utils.py


test_all: update_local test_qcd_utils test_gaugetheoryshower test_parton test_shower_algorithm test_vector_utils
