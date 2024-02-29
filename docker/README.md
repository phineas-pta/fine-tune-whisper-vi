fine-tune using LoRA based on `vinai/PhoWhisper-large`

deploy on AWS EC2

below is just my memory aid to run docker locally

```bash
docker build --platform=linux/amd64 --tag=tesstt .
docker run --rm -dit tesstt
# docker login -u <registry-user> -p <registry-password> <registry-address>
docker tag <image-identifier> <registry-address>/<image-identifier>:<tag-name>
docker push <registry-address>/<image-identifier>:<tag-name>
```
