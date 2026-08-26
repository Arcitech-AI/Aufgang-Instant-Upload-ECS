import json
import os
import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ecs_client = boto3.client("ecs")

# ECS Config
ECS_CLUSTER = os.environ.get(
    "ECS_CLUSTER",
    "ECS-Cluster-For-PROD-Above20MB-Vectorization"
)

ECS_TASK_DEFINITION = os.environ.get(
    "ECS_TASK_DEFINITION",
    "ECS-Task-For-PROD-Above20MB-Vectorization:4"
)

ECS_LAUNCH_TYPE = os.environ.get(
    "ECS_LAUNCH_TYPE",
    "FARGATE"
)

ECS_CONTAINER_NAME = os.environ.get(
    "ECS_CONTAINER_NAME",
    "PROD-Above20MB-Vectorization-container"
)

ECS_SUBNETS = os.environ.get(
    "ECS_SUBNETS",
    "subnet-0a248ae8378e59f8f"
).split(",")

ECS_SECURITY_GROUPS = os.environ.get(
    "ECS_SECURITY_GROUPS",
    "sg-03d567864e7e1e9d7,sg-021bfa1990fc5b26f"
).split(",")

ECS_ASSIGN_PUBLIC_IP = os.environ.get(
    "ECS_ASSIGN_PUBLIC_IP",
    "ENABLED"
)


def lambda_handler(event, context):

    logger.info("Received SQS Event")
    logger.info(json.dumps(event, default=str))

    records = event.get("Records", [])

    if not records:
        logger.warning("No SQS records found")
        return {
            "statusCode": 400,
            "message": "No records found"
        }

    try:

        # Convert complete SQS payload to JSON string
        sqs_payload = json.dumps(event)

        logger.info("Starting ECS task...")

        response = ecs_client.run_task(
            cluster=ECS_CLUSTER,
            launchType=ECS_LAUNCH_TYPE,
            taskDefinition=ECS_TASK_DEFINITION,

            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": ECS_SUBNETS,
                    "securityGroups": ECS_SECURITY_GROUPS,
                    "assignPublicIp": ECS_ASSIGN_PUBLIC_IP
                }
            },

            overrides={
                "containerOverrides": [
                    {
                        "name": ECS_CONTAINER_NAME,
                        "environment": [
                            {
                                "name": "SQS_PAYLOAD",
                                "value": sqs_payload
                            }
                        ]
                    }
                ]
            }
        )

        failures = response.get("failures", [])
        if failures:
            logger.error("ECS task FAILED to start: %s", json.dumps(failures, default=str))
            return {
                "statusCode": 500,
                "message": "ECS task failed to start",
                "failures": json.loads(json.dumps(failures, default=str))
            }

        logger.info("ECS task started successfully")
        logger.info(json.dumps(response, default=str))

        return {
            "statusCode": 200,
            "message": "ECS task triggered successfully",
            "tasks": json.loads(json.dumps(response.get("tasks", []), default=str))
        }

    except Exception as e:
        logger.exception("Error while starting ECS task")

        return {
            "statusCode": 500,
            "message": str(e)
        }