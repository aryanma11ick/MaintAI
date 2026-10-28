# CI/CD setup

The workflow in `workflows/ci-cd.yml` runs tests for pull requests and pushes,
publishes a Docker image to GitHub Container Registry, and deploys `main` to EC2.

On the EC2 host, copy this repository to the directory stored in `EC2_APP_DIR`
and make sure Docker Compose is installed. The host must contain both
`compose.yaml` and `compose.prod.yaml`. The named `maintai-data` volume is kept
across deployments, so model state and SQLite data are not recreated by a deploy.

Configure these GitHub repository secrets:

- `EC2_HOST`: EC2 public DNS name or IP
- `EC2_USER`: normally `ubuntu`
- `EC2_SSH_KEY`: complete private SSH key, including BEGIN/END lines
- `EC2_APP_DIR`: absolute application directory, for example `/home/ubuntu/maintai`
- `GHCR_USERNAME`: GitHub username or machine-user name
- `GHCR_TOKEN`: a read-only GitHub token with `read:packages`

The GHCR package must be visible to the EC2 token. For a private repository,
configure the package/repository access in GitHub before the first deployment.
Keep ports 8000 and 8501 bound to localhost and put Nginx or an AWS load
balancer in front of Streamlit for HTTPS.
