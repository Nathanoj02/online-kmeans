# Signals
Project for Signals, Image &amp; Video, Master's Degree in Artificial Intelligence Systems

## Setup
Clone repository
```
git clone git@github.com:Nathanoj02/signals.git
```

Update submodule
```
git submodule update --init --recursive
```

Cpp setup
```
mkdir build && cd build
```
```
cmake ..
```
```
make -j
```

## Arguments
Operation (image, video)
```
-o image
```

Source image / video path
```
-s path
```

Destination image / video path
```
-d path
```

### Optional arguments
k value (int)
```
-k value
```

Max tolerance / error value (float)
```
-e value
```

Algorithm (only for image) (python, cpp, cuda, cudashared)
```
-a algorithm
```
