# Above service buildspecs

Configure each AWS CodeBuild project or pipeline action with the matching path:

| Environment | Buildspec path | Image tag |
| --- | --- | --- |
| Development | `above/cicd/dev/buildspec.yml` | `DEV` |
| Quality assurance | `above/cicd/qat/buildspec.yml` | `QAT` |
| Pre-production | `above/cicd/pre/buildspec.yml` | `PRE` |
| Production | `above/cicd/prod/buildspec.yml` | `PROD` |

The build context is explicitly rooted at `$CODEBUILD_SRC_DIR/above`. Keep secrets
outside these files and provide them through IAM, Parameter Store, or Secrets Manager.
