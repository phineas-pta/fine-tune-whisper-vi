FROM huggingface/peft-gpu

RUN echo -e '#!/bin/bash\nsource activate peft\npython "$@"\n' > /entrypoint.sh &&\
	chmod +x /entrypoint.sh

WORKDIR /workspace

ENV XDG_CACHE_HOME=/workspace/cache

COPY train.py /workspace/

# RUN apt-get update && source activate peft && pip install --no-cache-dir 'datasets[audio]'

ENTRYPOINT ["/entrypoint.sh"]

# CMD ["train.py"]
