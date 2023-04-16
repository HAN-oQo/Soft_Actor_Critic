#!/bin/bash
useradd -s /bin/bash -m ${USER_NAME}
usermod -u ${USER_ID} ${USER_NAME}
groupadd -g ${GROUP_ID} ${GROUP_NAME}
usermod -g ${GROUP_NAME} ${USER_NAME}

export HOME=/home/${USER_NAME}
echo cd ${WORKDIR} >> ${HOME}/.bashrc

chown -R ${USER_NAME}:${USER_NAME} /app

if [[ -z  "$COMMAND" ]]; then
  su ${USER_NAME} -s /bin/bash
else
  su ${USER_NAME} -c "${COMMAND}"
fi