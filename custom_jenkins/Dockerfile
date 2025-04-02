# Use the Jenkins image as the base image
FROM jenkins/jenkins:lts

# Switch to root user to install dependencies
USER root

# Install prerequisites and Docker
RUN apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | tee /etc/apt/trusted.gpg.d/docker.asc && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable" | tee /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    apt-get clean

# Add Jenkins user to the Docker group (handle existing group)
RUN getent group docker || groupadd docker && \
    usermod -aG docker jenkins

# Create the Docker directory and volume for DinD
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

# Switch back to the Jenkins user
USER jenkins
