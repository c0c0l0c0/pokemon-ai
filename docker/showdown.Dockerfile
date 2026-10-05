# Local Pokémon Showdown server for training, the same as running
# `node pokemon-showdown start --no-security` from a checkout (src/utils/server.py).
FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinned so formats and mechanics don't change under a run, bump on purpose
ARG SHOWDOWN_REF=3de1c70a62e63853fc84e084c7fac65fa9a34003

WORKDIR /showdown
RUN git init -q . \
    && git remote add origin https://github.com/smogon/pokemon-showdown.git \
    && git fetch -q --depth 1 origin "$SHOWDOWN_REF" \
    && git checkout -q FETCH_HEAD \
    && npm ci --omit=dev \
    && node build \
    && cp config/config-example.js config/config.js

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=10 \
    CMD node -e "require('net').connect(8000, '127.0.0.1').on('connect', () => process.exit(0)).on('error', () => process.exit(1))"

CMD ["node", "pokemon-showdown", "start", "--no-security"]
