# safe-perception
Perception with uncertainty quantification

## Getting Started
First, install `docker` with [Docker Compose](https://docs.docker.com/compose/install/).  
Then:
```
docker compose build
docker compose up
```

Open a shell in the container e.g.
```
docker ps
docker exec -it <container> bash
```

and navigate to the `src` directory
```
cd /root/src
```

You are now ready to execute code!

## Julia Setup

From inside the docker container
```
julia

# Activate the environment
] activate .

# Install the packages
] instantiate
```

