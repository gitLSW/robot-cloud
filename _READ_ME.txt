- For Docker Install: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html
- To start Docker Isaac:
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:2023.1.1



Omniverse Installation:
 0. sudo apt install libfuse2
 1. Install Omniverse: https://developer.nvidia.com/isaac-sim
 2. In Omniverse install:
      - Isaac Sims
      - Nucleus -> Launch Local Server
      - Streaming Client: https://docs.omniverse.nvidia.com/streaming-client/latest/user-manual.html
      - Recommended: Nucleus Navigator
 6. To Launch:
    1. First Launch Omniverse and connect to your local nucleus server.
    	- If you forgot your login: 
    		Uninstall Nucleus
    		Delete the folders below:
        		~/.local/share/ov/data/Auth
        		~/.local/share/ov/data/server
        		~/.local/share/ov/data/tagging_service
    		Re-install Nucleus
      - To find the Isaac Assets navigate using the Nucleus Navigator to:
            localhost/NVIDIA/Assets/Isaac
    2. Run: bash _setup.sh
    3.Start your Isaac Python Scripts: isaac PATH/TO/CODE.py


Convenience:
 - Set the Interpreter in VS to ~/.local/share/ov/pkg/isaac_sim-2023.1.0/python.sh
 - Add this to ~/.bashrc:
alias isaac="~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh"
alias isaacDir="open ~/.local/share/ov/pkg/isaac_sim-2023.1.1"
alias isaacLog="open ~/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/logs/Kit/omni.isaac.sim.python.gym"
    
    Now luanch with: isaac PATH/TO/CODE


Info:
 - Isaac Defult Location:
   open ~/.local/share/ov/pkg/isaac_sim-2023.1.0
 - Isaac Kit: ~/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/exts
 - Isaac Omni Extensions: ~/.local/share/ov/pkg/isaac_sim-2023.1.1/exts
 - Log Location:
   open ~/.local/share/ov/pkg/isaac_sim-2023.1.0/kit/logs/Kit/Isaac-Sim/2023.1/
 - Isaac Assets Import
    - Remote: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0
    - Local: omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.0
