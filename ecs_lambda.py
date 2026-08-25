import boto3
import json

ecs = boto3.client("ecs")

def handler(event, context):
    for record in event.get("Records", []):
        ecs.run_task(
            cluster="your-cluster-name",
            taskDefinition="your-task-def",
            launchType="FARGATE",
            networkConfiguration={...},
            overrides={
                "containerOverrides": [{
                    "name": "your-container-name",
                    "command": ["python", "app.py", record["body"]]  # pass payload as CLI arg
                }]
            }
        )