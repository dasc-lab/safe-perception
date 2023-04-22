# safe-perception
Perception with uncertainty quantification

## Getting Started
First, install `docker compose` (NOT `docker-compose`) according to the instructions [here](https://docs.docker.com/compose/install/). 

Recommended: To be able to run Docker without typing `sudo`, follow the instructions [here](https://docs.docker.com/engine/install/linux-postinstall/).

Then:
```
docker compose build
docker compose up
```

You may run docker compose in the background, e.g. 
```
docker compose up &
```
or simply leave the docker window open and switch to a new tab to proceed. 

You can list currently running docker containers with 
```
docker ps
```

Open a shell in the container with 

```
docker exec -it <container name> bash
```

If you have set up docker to work without `sudo`, you can use the Tab key to auto-complete container names. 

You are now ready to execute code! 

## Julia Setup

From inside the docker container in the `/root` folder, run: 
```
julia src/init.jl
```

This will install all the packages needed (may take some time). 

When running code, either run `init.jl` to set up a new REPL, or 
```
cd src/

julia
```

And then inside the Julia REPL, run 
```
# Activate the environment as defined by the relevant .toml files
] activate .

# Install/load the packages
] instantiate
```
