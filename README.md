# G.O.A.T. (Getting Over All Things)
Wheeled Bipedal Robot(WBR) Project by *Hansu Kim*, *Hongryeol Yoon*, *Sangjun Moon*, *Haechan Lee* in AISL Chung-ang university   

## Ennvironment setup
GOAT is based on Reinforcement Learning, using Isaac Sim, Isaac Lab, Pytorch.   
All Learning environment is wrapped by Docker container.   

### 1. Host computer setup  

Berfore Build, follow the **Container Setup** instruction at [Isaac sim Docker Docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_container.html)   
Check **Docker**, **Nvidia container toolkit**, **cuda** is installed.
>Make sure that host computer's **cuda version** is **at least 12.8**   

### 2. Build Dockerfile

```bash
cd ${PATH_TO_GOAT_REPO}
sudo docker build -t goat_rlenv:isaaclab .   # build Dockerfile
```
### 3. Run a container
```bash
docker run --name goat --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    goat_rlenv:isaaclab    # Run a container
```

## API references
Open source API Docs URL
1. **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/api/index.html)**
2. **[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/index.html)**
3. **[Omni physx](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView)**