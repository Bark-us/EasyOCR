torchserve --start --ncs --model-store model_store
sleep 1
curl -X POST "localhost:8081/models?url=craft.mar&initial_workers=1&"
curl -X POST "localhost:8081/models?url=text.mar&batch_size=32&initial_workers=1&max_batch_delay=10"
