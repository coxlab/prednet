right_after_pull_docker_image=$(date +%s)
cat /etc/os-release || echo "cat /etc/os-release failed."
lsb_release -a || echo "lsb_release -a failed."
hostnamectl || echo "hostnamectl failed."
uname -r || echo "uname -r failed."
echo $(whoami)
echo $USER
if [ -z ${ETC_ENVIRONMENT_LOCATION+ABC} ]; then echo "ETC_ENVIRONMENT_LOCATION is unset, so assuming you do not need environment variables set."; else
if [ -z ${ETC_ENVIRONMENT_LOCATION} ]; then echo "ETC_ENVIRONMENT_LOCATION is set to the empty string; I hope you know why, because I certainly do not."; fi
echo "ETC_ENVIRONMENT_LOCATION = $ETC_ENVIRONMENT_LOCATION"
mkdir --parents ~/.ssh
echo "PasswordAuthentication=no" >> ~/.ssh/config
echo $SSH_PRIVATE_DEPLOY_KEY > SSH.PRIVATE.KEY
wget $ETC_ENVIRONMENT_LOCATION --output-document environment.sh --no-clobber || (wget --help && wget --proxy off $ETC_ENVIRONMENT_LOCATION --output-document environment.sh --no-clobber) || curl --verbose $ETC_ENVIRONMENT_LOCATION --output environment.sh || scp -i SSH.PRIVATE.KEY $ETC_ENVIRONMENT_LOCATION environment.sh
rm SSH.PRIVATE.KEY
cat environment.sh
SAVED_PATH=$PATH
set -o allexport
. ./environment.sh
set +o allexport
PATH=$SAVED_PATH
fi
if [ -z ${SSH_PRIVATE_DEPLOY_KEY+ABC} ]; then echo "SSH_PRIVATE_DEPLOY_KEY is unset, so assuming you do not need SSH set up."; else
if [ ${#SSH_PRIVATE_DEPLOY_KEY} -le 5 ]; then echo "SSH_PRIVATE_DEPLOY_KEY looks far too short, something is wrong"; fi
apk add openssh-client || apt-get install --assume-yes openssh-client || (apt-get update && apt-get install --assume-yes openssh-client)  || echo "Failed to install openssh-client; proceeding anyway to see if this image has its own SSH."
echo "adding openssh-client took $(( $(date +%s) - right_after_pull_docker_image)) seconds"
eval $(ssh-agent -s)
echo "$SSH_PRIVATE_DEPLOY_KEY" | tr -d '\r' | ssh-add -
echo "Added the private SSH deploy key with public fingerprint $(ssh-add -l)"
echo "WARNING! If you use this script to build a Docker image (rather than just run tests), make sure to delete the deploy key with ssh-add -D after installing the relevant repos."
mkdir --parents ~/.ssh
echo "# github.com:22 SSH-2.0-babeld-f345ed5d\n" >> ~/.ssh/known_hosts
echo "github.com ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==\n" >> ~/.ssh/known_hosts
echo "# gitlab.com:22 SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.8\n" >> ~/.ssh/known_hosts
echo "gitlab.com ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCsj2bNKTBSpIYDEGk9KxsGh3mySTRgMtXL583qmBpzeQ+jqCMRgBqB98u3z++J1sKlXHWfM9dyhSevkMwSbhoR8XIq/U0tCNyokEi/ueaBMCvbcTHhO7FcwzY92WK4Yt0aGROY5qX2UKSeOvuP4D6TPqKF1onrSzH9bx9XUf2lEdWT/ia1NEKjunUqu1xOB/StKDHMoX4/OKyIzuS0q/T1zOATthvasJFoPrAjkohTyaDUz2LN5JoH839hViyEG82yB+MjcFV5MU3N1l1QL3cVUCh93xSaua1N85qivl+siMkPGbO5xR/En4iEY6K2XPASUEMaieWVNTRCtJ4S8H+9\n" >> ~/.ssh/known_hosts
fi
if [ -z ${SERVERS_TO_WHITELIST_FOR_SSH+ABC} ] || [ -z ${SSH_PRIVATE_DEPLOY_KEY+ABC} ]; then echo "SERVERS_TO_WHITELIST_FOR_SSH and SSH_PRIVATE_DEPLOY_KEY are not both set, so assuming you do not need any servers whitelisted for SSH."; else
echo "SERVERS_TO_WHITELIST_FOR_SSH = $SERVERS_TO_WHITELIST_FOR_SSH"
mkdir --parents ~/.ssh
ssh-keyscan -t rsa $SERVERS_TO_WHITELIST_FOR_SSH >> ~/.ssh/known_hosts
fi
if command -v conda; then echo "command finds conda"; else echo "command does not find conda"; fi
if [ -d /opt/conda ]; then
CONDA_DIR=/opt/conda
PATH=$CONDA_DIR/bin:$PATH
if [ "$CONDA_DEFAULT_ENV" = "test-env" ]; then echo "This image already has test-env activated."; else
conda env list
fi
if [ "$CONDA_DEFAULT_ENV" = "test-env" ] || source activate test-env; then true; else echo "No conda env named test-env was found, so not activating any particular env."; fi ; else echo "/opt/conda was not found on this container" ; fi
if command -v python; then python -m pip install --upgrade pip; fi
if [ -z ${PROXY_CA_PEM+ABC} ]; then echo "PROXY_CA_PEM is unset, so assuming you do not need a merged CA certificate set up."; else
right_before_pull_cert=$(date +%s)
if [ ${#PROXY_CA_PEM} -ge 1024 ]; then
echo "The PROXY_CA_PEM filename looks far too long, did you set it as a Variable instead of a File?"
echo "$PROXY_CA_PEM" > tmp-proxy-ca.pem
PROXY_CA_PEM=tmp-proxy-ca.pem ; fi
if command -v python; then
python --version
python -c "import contextlib; contextManager = contextlib.suppress(AttributeError); contextManager.__enter__(); import pip._vendor.requests; contextManager.__exit__(None,None,None); from pip._vendor.requests.certs import where; print(where())"
cat $(python -c "import contextlib; contextManager = contextlib.suppress(AttributeError); contextManager.__enter__(); import pip._vendor.requests; contextManager.__exit__(None,None,None); from pip._vendor.requests.certs import where; print(where())") ${PROXY_CA_PEM} > bundled.pem
ls bundled.pem
export REQUESTS_CA_BUNDLE="${PWD}/bundled.pem"
echo "REQUESTS_CA_BUNDLE found at $(ls $REQUESTS_CA_BUNDLE)"
echo "Merging the certificate bundle took $(( $(date +%s) - right_before_pull_cert)) seconds total"
fi
fi
python3 --version || echo "python3 is not found by that name."
if command -v jupyter; then
pip install ipykernel
python -m ipykernel install
pip install ipywidgets
fi
echo "before_script took $(( $(date +%s) - right_after_pull_docker_image)) seconds total"
