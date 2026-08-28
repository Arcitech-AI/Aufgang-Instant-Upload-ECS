# Below service buildspecs

Configure each AWS CodeBuild project or pipeline action with the matching path:

| Environment | Buildspec path | Image tag |
| --- | --- | --- |
| Development | `below/cicd/dev/buildspec.yml` | `DEV` |
| Quality assurance | `below/cicd/qat/buildspec.yml` | `QAT` |
| Pre-production | `below/cicd/pre/buildspec.yml` | `PRE` |
| Production | `below/cicd/prod/buildspec.yml` | `PROD` |

The build context is explicitly rooted at `$CODEBUILD_SRC_DIR/below`. These Lambda
builds require Docker Buildx and a CodeBuild environment with Docker support. Keep
secrets outside these files and provide them through IAM, Parameter Store, or Secrets Manager.
