fine-tune using LoRA based on https://huggingface.co/vinai/PhoWhisper-large

remove VIVOS & CommonVoice because already included in above checkpoint

also remove FLEURS so post-training evaluation will be on out-of-distribution data

objective: deploy on AWS EC2

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

storage: at least 32 GB

network settings: select any security group

advanced settings: user data: install docker … (somehow doesn’t work)

edit security groups: add new inbound rules: type SSH + source my IP

connect to instance using SSH

install docker:
```bash
sudo su
yum update -y
yum install docker -y
service docker start
usermod -a -G docker ec2-user
```
exit then reconnect
```bash
docker login --username="someuser" --password="asdfasdf"
docker run -it --rm --gpus=all -e HF_TOKEN=███ -v ~/.cache:/workspace/cache -v ~/coder/whisper:/workspace/my-whisper-lora <username>/<my-repo>:<tag> train.py --help
```
