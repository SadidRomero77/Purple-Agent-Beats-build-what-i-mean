FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser --disabled-password --gecos "" agent
USER agent
WORKDIR /home/agent

COPY --chown=agent:agent pyproject.toml README.md ./
COPY --chown=agent:agent src src

RUN uv sync --no-dev

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9018"]

EXPOSE 9018
