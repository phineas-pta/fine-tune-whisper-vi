fine-tune using LoRA based on https://huggingface.co/vinai/PhoWhisper-large

remove VIVOS & CommonVoice because already included in above checkpoint

also remove FLEURS so post-training evaluation will be on out-of-distribution data

objective: deploy on AWS EC2 (instance with multiple GPU)

#  my memory aid to run docker locally

```bash
docker build --platform=linux/amd64 --tag=tesstt .
docker run -it --rm --gpus=all -e HF_TOKEN=███ -v ~/.cache:/workspace/cache -v ~/coder/whisper:/workspace/my-whisper-lora tesstt train.py --help

docker stop tesstt
docker images tesstt
docker rmi tesstt

# docker login -u <registry-user> -p <registry-password> <registry-address>
docker tag tesstt <username>/<my-repo>:<tag>
docker push <username>/<my-repo>:<tag>
```
result image: 17 GB

other approach: using docker compose
```bash
docker compose up -d
docker compose stop
docker compose down --volumes
```
additional things: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# my memory aid to launch an AWS EC2 instance

region: select nearest for best latency

choose Amazon Linux image, arch: `x86_64`

instance type: `t2.micro` (free tier) or with nvidia gpu: see https://aws.amazon.com/ec2/instance-types/ `>>>` Accelerated Computing

key pair: `.pem` file for SSH: remember to edit permission to something like `chmod 400` or on windows see https://superuser.com/a/1329702/990893

storage: select “Advanced”:
- EBS volume (for OS): at least 32 GB
- instance store volume (for data and cache - depends on instance type): auto-mounted at `/mnt`

network settings: select any security group

~~advanced settings: user data: install docker …~~ (somehow doesn’t work)

edit security groups: add new inbound rules: type SSH + source my IP

connect to instance using SSH

install docker:
```bash
sudo su
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo > /etc/yum.repos.d/nvidia-container-toolkit.repo
yum update -y
yum install -y vim docker nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
vim /etc/docker/daemon.json  # add: {"data-root": "/path/to/your/new/docker/root"}
usermod -a -G docker ec2-user
systemctl start docker

mkdir -p ~/.vim/pack/dist/start
cd ~/.vim/pack/dist/start
wget https://github.com/vim-airline/vim-airline/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
mv vim-airline-master vim-airline

tee ~/.vimrc <<EOT
set visualbell
set noerrorbells
set number
set encoding=UTF-8
set tabstop=4
set shiftwidth=4
let g:airline_powerline_fonts=1
EOT
```
exit then copy `train.py` to aws (*e.g.* `~/whisper`) then reconnect
```bash
docker login --username="someuser" --password="asdfasdf"
docker run --gpus=all --rm \
	-e OMP_NUM_THREADS=1 -e HF_TOKEN=███ \
	-v ~/whisper:/workspace \
	<username>/<my-repo>:<tag> \
	python train.py -total-steps 30 -batch-size 4
```
monitor VRAM usage: `watch nvidia-smi`

# memory allocation: vram usage &amp; batch size

use case: single node - multiple GPU

default config with `transformers` trainer: vertical model parallelism (assigning specific layers to specific GPUs)

attempt to run distributed data parallelism (assigning specific batches to specific GPUs): but error: each GPU get different batch tensor size
