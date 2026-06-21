# File: Dockerfile
# PURPOSE: Custom container for HF Spaces that pre-installs Node.js MCP
#          servers globally, so npx doesn't reinstall them (and print
#          noisy install logs to stdout) on every agent run.
# WHERE THIS RUNS: Hugging Face Spaces build step (replaces default
#                   Gradio SDK image)

FROM python:3.13-slim

# Install Node.js (needed for npx-based MCP servers)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Pre-install MCP npm servers GLOBALLY so npx finds them instantly
# instead of downloading + printing install logs on every run
RUN npm install -g \
    @modelcontextprotocol/server-brave-search \
    @modelcontextprotocol/server-filesystem \
    mcp-server-sqlite-npx

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure the folders the agent depends on exist at runtime
RUN mkdir -p /app/output /app/templates /app/data /app/servers

EXPOSE 7860

CMD ["python", "app.py"]