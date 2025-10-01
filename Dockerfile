# syntax=docker/dockerfile:1.6

# Isaac Sim 4.5 (Ubuntu 22.04 / Python 3.10)
ARG ISAAC_SIM_TAG=4.5.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_TAG}

# Non-interactive + headless defaults
SHELL ["/bin/bash", "-lc"]
ENV TERM=xterm-256color \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul LANG=C.UTF-8 \
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
    HEADLESS=1

#======= ROOT USER SETUP =======#

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates sudo locales build-essential cmake ninja-build pkg-config vim \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 ko_KR.UTF-8

# Isaac Sim python setup
RUN printf '%s\n' '#!/usr/bin/env bash' \
                 'exec /isaac-sim/python.sh "$@"' \
    > /usr/local/bin/python \
    && chmod +x /usr/local/bin/python \
    && /usr/local/bin/python --version

# Create user
ARG USERNAME=goat
RUN useradd -m -s /bin/bash "${USERNAME}" \
    && usermod -aG sudo "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add user's local bin to PATH
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Grant user permissions for Isaac Sim system dirs
RUN mkdir -p /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs \
    && chown -R ${USERNAME}:${USERNAME} /isaac-sim/kit/cache /isaac-sim/kit/data /isaac-sim/kit/logs

# System-wide python packages
RUN python -m pip install --upgrade pip setuptools wheel
ARG CUDA_VERSION=cu128
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Create and set ownership for user's workspace
RUN mkdir -p /home/${USERNAME}/workspace && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}


#======= GOAT USER SETUP =======#

# Switch to user
USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

# Clone Isaac Lab repo
RUN git clone --depth=1 https://github.com/isaac-sim/IsaacLab.git

# Modify Isaac Lab installation script
SHELL ["/bin/bash", "-lc"]
RUN <<'BASH'
set -euo pipefail
replace_func_body() {
  local file="$1" fn="$2"
  local bfile; bfile="$(mktemp)"
  cat > "$bfile"
  awk -v fn="$fn" -v BF="$bfile" '
    function slurp(f,  t,s){ s=""; while((getline t < f)>0) s=s t ORS; close(f); return s }
    BEGIN{ inside=0; depth=0; header_seen=0; BODY=slurp(BF) }
    {
      if(!inside){
        pat1="^[ \t]*(function[ \t]+)?(" fn ")[ \t]*(\(\))?[ \t]*\{[ \t]*$"
        pat2="^[ \t]*(function[ 	]+)?(" fn ")[ \t]*(\(\))?[ \t]*$"
        if ($0 ~ pat1){ inside=1; depth=1; print $0; printf "%s", BODY; print "}"; next }
        if ($0 ~ pat2){ header_seen=1; header_line=$0; next }
        print; next
      } else {
        s=$0; ob=gsub(/\{/,"{",s); s=$0; cb=gsub(/\}/,"}",s); depth+=ob-cb
        if (depth<=0){ inside=0 }
        next
      }
    }
    header_seen && /^\s*{\s*$/ { inside=1; depth=1; print header_line; print $0; printf "%s", BODY; print "}"; header_seen=0; next }
    header_seen { print header_line; print; header_seen=0 }
  ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
  rm -f "$bfile"
}
FILE="/home/goat/workspace/IsaacLab/isaaclab.sh"

# 1) Exchange extract_python_exe()
BODY1=$(cat <<'EOF'
    local python_exe=/isaac-sim/python.sh

    if [ ! -f "${python_exe}" ]; then
            # note: we need to check system python for cases such as docker
            # inside docker, if user installed into system python, we need to use that
            # otherwise, use the python from the kit
            if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
                local python_exe=$(which python)
            fi
    fi
    # check if there is a python path available
    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] Unable to find any Python executable at path: '${python_exe}'" >&2
        echo -e "	This could be due to the following reasons:" >&2
        echo -e "	1. Conda or uv environment is not activated." >&2
        echo -e "	2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "	3. Python executable is not available at the default path: /isaac-sim/python.sh" >&2
        exit 1
    fi
    # return the result
    echo ${python_exe}
EOF
)
echo "${BODY1}" | replace_func_body "$FILE" "extract_python_exe"

# 2) Exchange extract_isaacsim_path()
BODY2=$(cat <<'EOF'
    # Use the sym-link path to Isaac Sim directory
    local isaac_path=/isaac-sim
    # If above path is not available, try to find the path using python
    if [ ! -d "${isaac_path}" ]; then
        # Use the python executable to get the path
        local python_exe=$(extract_python_exe)
        # Retrieve the path importing isaac sim and getting the environment path
        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            local isaac_path=$(${python_exe} -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH',''))")
        fi
    fi
    # check if there is a path available
    if [ ! -d "${isaac_path}" ]; then
        echo -e "[ERROR] Unable to find the Isaac Sim directory: '${isaac_path}'" >&2
        echo -e "	This could be due to the following reasons:" >&2
        echo -e "	1. Conda environment is not activated." >&2
        echo -e "	2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "	3. Isaac Sim directory is not available at the default path: /isaac-sim" >&2
        exit 1
    fi
    # return the result
    echo ${isaac_path}
EOF
)
echo "${BODY2}" | replace_func_body "$FILE" "extract_isaacsim_path"
BASH

# Install Isaac Lab
RUN find ./IsaacLab/source -maxdepth 2 -type d -name "*.egg-info" -print -exec rm -rf {} + || true
RUN bash ./IsaacLab/isaaclab.sh --install

# Clone GOAT repo
RUN git clone https://github.com/OKsusu1411/GOAT.git

# Launch bash
WORKDIR /home/${USERNAME}/workspace/GOAT
CMD ["bash"]