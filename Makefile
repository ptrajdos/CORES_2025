ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))
DATAFILE=${ROOTDIR}/tsnre_windowed.tar.xz
DATAFILEID=15D-PUeOIHQXyJMhaoDGOxmSfAayMyBgb
DATADIR=${ROOTDIR}/data
VENV_SUBDIR=${ROOTDIR}/venv
EXPERIMENT_SUBDIR=${ROOTDIR}/dexterous_bioprosthesis_2021_raw_datasets_framework_experiments
LOGFILE=${ROOTDIR}/install.log

PYTHON=python
PIP=pip
CURL=curl
TAR=tar

.PHONY: all clean

.NOTPARALLEL: run

create_env: venv data

clean:
	rm -rf ${VENV_SUBDIR}


run: create_env
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${EXPERIMENT_SUBDIR}/channel_nb_weights1.py

venv:
	${PYTHON} -m venv ${VENV_SUBDIR}
	. ${VENV_SUBDIR}/bin/activate; ${PIP} install -e . --log ${LOGFILE}

data:
	mkdir -p ${DATADIR}
	${CURL} -L -o ${DATAFILE} "https://drive.usercontent.google.com/download?id=${DATAFILEID}&export=download&authuser=1&confirm=t"
	${TAR} -xvf ${DATAFILE} --directory ${DATADIR}
	
