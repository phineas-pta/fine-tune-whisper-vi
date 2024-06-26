fine-tune using LoRA based on https://huggingface.co/vinai/PhoWhisper-large (or `medium`)

remove VIVOS & CommonVoice because already included in above checkpoint

also remove FLEURS so post-training evaluation will be on out-of-distribution data

objective: deploy on AWS EC2 (instance with multiple GPU)

# run locally without docker

```
pip install torch torchaudio --find-links=https://download.pytorch.org/whl/torch_stable.html
pip install jiwer transformers "datasets[audio]" accelerate peft bitsandbytes
```
*e.g.* GPU with 8GB VRAM (scripts auto change number of gradient accumulation steps to have effective batch size at least 8)

```bash
python download_data.py
python train.py -pretrained-model vinai/PhoWhisper-medium -batch-size 4 -num-steps 11000 -save-path ./save-medium  # 3 s/step
python train.py -pretrained-model vinai/PhoWhisper-large  -batch-size 2 -num-steps 11000 -save-path ./save-large   # 7 s/step
python evaluate_wer.py -save-path ./save-medium -batch-size 16

python -m tensorboard.main --logdir ./save-medium/runs/…
```

# my memory aid to run docker locally

```bash
docker build --platform=linux/amd64 --tag=horimiyasanxmiyamurakun/wisuperuro .
docker run -it --rm --gpus=all -e HF_TOKEN=███ -v ~/.cache:/cache -v ~/coder/whisper:/workspace horimiyasanxmiyamurakun/wisuperuro python train.py --help

# docker login -u <user> -p <password>
docker push horimiyasanxmiyamurakun/wisuperuro

docker images
docker rmi horimiyasanxmiyamurakun/wisuperuro
```
result image: 17 GB, available at https://hub.docker.com/repository/docker/horimiyasanxmiyamurakun/wisuperuro

other approach: using docker compose
```bash
docker compose up -d
docker compose stop
docker compose down --volumes
```

# my memory aid to launch an AWS EC2 instance

## setup

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

mount S3 storage (to save checkpoint if use spot instance): https://github.com/jomyg/Mount-S3-Bucket-on-amazon-linux-using-S3FS

connect to instance using SSH

## install docker

```bash
mkdir -p ~/.vim/pack/dist/start
cd ~/.vim/pack/dist/start
wget https://github.com/vim-airline/vim-airline/archive/refs/heads/master.zip
unzip master.zip
rm master.zip
mv vim-airline-master vim-airline
echo -e 'set visualbell\nset noerrorbells\nset number\nset encoding=UTF-8\nset tabstop=4\nset shiftwidth=4\nlet g:airline_powerline_fonts=1' > ~/.vimrc

echo 'ACCESS_KEY_ID:SECRET_ACCESS_KEY' > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs

sudo su  # all commands below as root
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo > /etc/yum.repos.d/nvidia-container-toolkit.repo
yum update -y
yum install -y vim s3fs docker nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
vim /etc/docker/daemon.json  # add: {"data-root": "/path/to/your/new/docker/root"}  # default is `/var/lib`
usermod -a -G docker ec2-user
systemctl start docker

mkdir /mnt/workingspace
mkfs -t xfs /dev/nvme1n1  # check device name with `lsblk -f`
mount /dev/nvme1n1 /mnt/workingspace

mkdir /mnt/s3
s3fs <your-s3-bucket-name> /mnt/s3 -o passwd_file=~/.passwd-s3fs -o url=https://s3-<aws_region>.amazonaws.com -o allow_other
```
exit then copy `train.py` to aws (*e.g.* `~/whisper`) then reconnect

## run docker

```bash
docker pull horimiyasanxmiyamurakun/wisuperuro
docker ps -a
# do not enable `--rm` to keep logs
yolo() { tmp="$1"; shift 1; docker run --gpus=all -e HF_TOKEN=███ -v ~/.cache:/cache -v /mnt/s3/whisper:/workspace --name "$tmp" horimiyasanxmiyamurakun/wisuperuro "$@"; }
yolo my-container-download python download_data.py
yolo my-container-train    python train.py -pretrained-model vinai/PhoWhisper-large -num-steps 30 -batch-size 4
yolo my-container-train-ddp \
	python -m accelerate.commands.launch \
	--multi_gpu --mixed_precision fp16 --num_machines 1 --num_processes 2 \
	train_ddp.py -pretrained-model vinai/PhoWhisper-large -num-steps 30 -batch-size 4
yolo my-container-evaluate python evaluate_wer.py
```
monitor VRAM usage: `watch nvidia-smi`

view logs: `docker logs <container_id>` or see `<docker-data-root>/docker/containers/<container id>/<container id>-json.log` (save logs to S3 storage)

remove stopped container: `docker rm $(docker ps -a -f status=exited -q)`
