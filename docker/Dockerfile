FROM huggingface/peft-gpu

RUN echo -e '#!/bin/bash\nsource activate peft\npython "$@"\n' > /entrypoint.sh &&\
	chmod +x /entrypoint.sh

WORKDIR /workspace

ENV XDG_CACHE_HOME=/workspace/cache

ENV PIP_ROOT_USER_ACTION=ignore

COPY train.py /workspace/

RUN apt-get update && source activate peft && pip install --no-cache-dir tensorboard

ENTRYPOINT ["/entrypoint.sh"]

# CMD ["train.py"]
