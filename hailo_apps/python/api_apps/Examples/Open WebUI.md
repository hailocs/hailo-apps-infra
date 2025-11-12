# Open WebUI
Open WebUI is an extensible, feature-rich, and user-friendly self-hosted AI platform designed to operate entirely offline: https://github.com/open-webui/open-webui.

Once hailo-ollama is up and running - please see [Ollama Readme](Ollama.md) - it's possible to use it with the popular Open WebUI.

## Prerequisites

### Installing Docker on Raspberry Pi

Follow these steps to install Docker on your Raspberry Pi:

#### Quick Installation (Recommended)

```bash
# Update package list
sudo apt-get update

# Install Docker using the convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Apply the group changes (or logout/login)
newgrp docker
```

#### Verify Installation

```bash
# Check Docker version
docker --version

# Test Docker with hello-world
docker run hello-world
```

#### Enable Docker to Start on Boot

```bash
sudo systemctl enable docker
```

**Note:** After adding your user to the docker group, you may need to log out and back in for the changes to take effect if `newgrp docker` doesn't work.

## Open WebUI Installation

Follow this guide: https://docs.openwebui.com/getting-started/quick-start

Download and run the slim variant:

```bash
docker pull ghcr.io/open-webui/open-webui:main-slim

docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main-slim
```

### Configure Open WebUI

1. Open your browser and navigate to the Open WebUI interface at http://localhost:3000

2. In **Settings → Admin Settings → Connections**, add the Hailo-Ollama API URL: 
   ```
   http://localhost:8000
   ```

3. Under the "Ollama API" section:
   - Set "Connection Type" to "Local"
   - Set "Auth" to "None"

4. Select the `qwen2:1.5b` model from the available models

