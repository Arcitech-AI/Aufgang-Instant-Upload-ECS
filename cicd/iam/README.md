# CodeBuild IAM policy

Recommended shared CodeBuild service-role name:

`CodeBuild-Aufgang-InstantUpload-ECR-Publisher`

Recommended customer-managed policy name:

`CodeBuild-Aufgang-InstantUpload-ECR-Push`

Attach `codebuild-ecr-push-policy.json` to the CodeBuild service role. The policy
permits authentication and image pushes only to the project's `above20mb` and
`below20mb` ECR repositories in account `615299760779`, Region `us-east-1`.

`ecr:GetAuthorizationToken` must use `Resource: "*"` because Amazon ECR does not
support repository-level resource scoping for that action. All layer and image
actions are restricted to the two repository ARNs.

This is only the ECR portion of the CodeBuild service-role policy. Retain the
project's existing least-privilege permissions for CloudWatch Logs and any S3,
CodeConnections, KMS, or artifact resources used by the build project.

For stronger production isolation, use separate roles per environment and ECR
repositories per environment. IAM cannot restrict `ecr:PutImage` to only a
particular image-tag value within a shared repository.
