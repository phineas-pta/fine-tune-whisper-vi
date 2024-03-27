FROM huggingface/peft-gpu

WORKDIR /workspace

ENV XDG_CACHE_HOME=/workspace/cache
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=yes

RUN apt-get update &&\
	apt-get install -y vim &&\
	source activate peft &&\
	pip install jiwer tensorboard boto3 &&\
	echo -e '#!/bin/bash\nsource activate peft\nexec "$@"\n' > /entrypoint.sh &&\
	chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
