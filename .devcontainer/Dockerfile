FROM ludwigai/ludwig-ray

# Copy library scripts to execute
RUN curl -o /tmp/common-debian.sh https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-debian.sh
RUN curl -o /tmp/github-debian.sh https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/github-debian.sh

USER root
# [Option] Install zsh
ARG INSTALL_ZSH="true"
# [Option] Upgrade OS packages to their latest versions
ARG UPGRADE_PACKAGES="true"
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    && apt-get purge -y imagemagick imagemagick-6-common \
    # Install common packages, non-root user
    && bash /tmp/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && bash /tmp/github-debian.sh \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# https://github.com/grpc/grpc/issues/25082#issuecomment-754565768
ENV GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
ENV GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

RUN pip install pre-commit black pylint flake8 mypy

# USER vscode
