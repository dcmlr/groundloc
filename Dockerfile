FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
SHELL ["/bin/bash", "-exo", "pipefail", "-c"]
# install ros
RUN apt-get update && apt-get install locales -y
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN apt install -y software-properties-common 
RUN add-apt-repository -y universe 
RUN apt-get update && apt-get install curl -y
RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
RUN dpkg -i /tmp/ros2-apt-source.deb
RUN apt-get update && apt-get install ros-dev-tools -y
RUN apt-get update && apt-get upgrade -y
RUN apt-get install ros-jazzy-desktop -y
RUN apt-get install python3-rasterio -y
RUN apt-get install ros-jazzy-rmw-cyclonedds-cpp -y
RUN apt-get install python3-pandas -y
RUN apt-get install python3-venv -y 
RUN rosdep init

# install torch
RUN apt-get install libnvinfer-bin libnvinfer-dev libnvinfer-plugin-dev -y
RUN apt-get install unzip -y
USER ubuntu
RUN mkdir /home/ubuntu/.local
WORKDIR /home/ubuntu/.local
RUN wget -q https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip
RUN unzip libtorch-shared-with-deps-2.8.0+cu129.zip
RUN rm libtorch-shared-with-deps-2.8.0+cu129.zip

# set up groundgrid
USER ubuntu
WORKDIR /home/ubuntu
RUN echo 'source /opt/ros/jazzy/setup.bash' | tee -a ~/.bashrc
RUN echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' | tee -a ~/.bashrc 
RUN echo 'LD_LIBRARY_PATH=/home/ubuntu/.local/libtorch/lib/:${LD_LIBRARY_PATH}' | tee -a ~/.bashrc 
RUN echo 'LD_LIBRARY_PATH=/home/ubuntu/ros/build/quatro/pmc/lib/:${LD_LIBRARY_PATH}' | tee -a ~/.bashrc 
RUN mkdir -p ros/src/groundloc
COPY ./ ros/src/groundloc/
RUN rosdep update

USER root
WORKDIR /home/ubuntu/ros
RUN chown -R ubuntu:ubuntu src/groundloc
RUN rosdep update
RUN rosdep install -i --from-path src --rosdistro jazzy -y
# fix compile error in ros point_cloud headers
RUN mv /home/ubuntu/ros/src/groundloc/groundloc/include/point_cloud.hpp /opt/ros/jazzy/include/pcl_ros/pcl_ros/point_cloud.hpp

USER ubuntu
WORKDIR /home/ubuntu/ros
RUN source /opt/ros/jazzy/setup.bash && colcon build
RUN echo 'source /home/ubuntu/ros/install/setup.bash' | tee -a ~/.bashrc
WORKDIR /home/ubuntu/ros/src/groundloc/groundloc/scripts
RUN mkdir venv
RUN python3 -m venv --system-site-packages ./venv
RUN source venv/bin/activate && pip install kiss-icp
WORKDIR /home/ubuntu/ros
ENTRYPOINT [ "/bin/bash", "-l" ]
