fine-tune using LoRA based on https://huggingface.co/vinai/PhoWhisper-large

remove VIVOS & CommonVoice because already included in above checkpoint

also remove FLEURS so post-training evaluation will be on out-of-distribution data

objective: deploy on AWS EC2

below is just my memory aid to run docker locally
```bash
docker build --platform=linux/amd64 --tag=tesstt .
docker run -it --rm --gpus=all -v ~/.cache:/workspace/cache -v ~/coder/whisper:/workspace/my-whisper-lora tesstt

docker stop tesstt
docker images tesstt
docker rmi tesstt

# docker login -u <registry-user> -p <registry-password> <registry-address>
docker tag <image-identifier> <registry-address>/<image-identifier>:<tag-name>
docker push <registry-address>/<image-identifier>:<tag-name>
```
other approach: using docker compose
```bash
docker compose up -d
docker compose stop
docker compose down --volumes
```
additional things: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
