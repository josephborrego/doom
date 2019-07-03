# doom
doom reinforcement learning project

downloads:
- preconfigured vm 
https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b

- IF MAC USER: if it's not working it's most likely because of security and privacy. Also make sure that you have VMWare Fusion to open the image.

steps:
- upgrade vm image os to most recent ubuntu downloads
https://www.tecmint.com/fix-unable-to-lock-the-administration-directory-var-lib-dpkg-lock/

- i was getting an error with apt-get upgrade because it was unable to lock the administrative directory. i had to delete the lock files like so

sudo rm /var/lib/dpkg/lock
sudo dpkg --configure -a
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo apt-get upgrade

https://www.tecmint.com/fix-unable-to-lock-the-administration-directory-var-lib-dpkg-lock/



- update pip3
https://stackoverflow.com/questions/38613316/how-to-upgrade-pip3

-install the dependicies 

# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 2 dependencies
sudo apt-get install python-dev python-pip
pip install numpy
# or install Anaconda 2 and add it to PATH

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Lua binding dependencies
sudo apt-get install liblua5.1-dev
# Lua shipped with Torch can be used instead, so it isn't needed if installing via LuaRocks

# Julia dependencies
sudo apt-get install julia
julia
julia> Pkg.add("CxxWrap")


- i had an import error issue because i didn't build it in the vizdoom directory before running basic.py

python setup.py build   # (or python3)

pip install -e .        # (or pip3)

https://github.com/openai/doom-py/issues/9




