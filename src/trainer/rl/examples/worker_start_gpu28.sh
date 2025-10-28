#!/bin/bash



export VLLM_ATTENTION_BACKEND=FLASH_ATTN
HEAD_IP=192.168.100.35  # 这里为head节点的IP，也就是机器A的IP
LOCAL_IP=192.168.100.37  # 这里为本机ip
PORT=8888  # 这里的port需要和前面的保持一致

# ray status
# 判断本机IP是否为Head节点的IP
if [ "$LOCAL_IP" == "$HEAD_IP" ]; then
    echo "本机 $LOCAL_IP 是Head节点，启动Head节点..."
    ray start --head --port=$PORT --min-worker-port=20122 --max-worker-port=20999
else
    echo "本机 $LOCAL_IP 是Worker节点，连接到Head节点 $HEAD_IP..."
    ray start --address=$HEAD_IP:$PORT --min-worker-port=20122 --max-worker-port=20999
fi