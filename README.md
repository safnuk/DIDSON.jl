# DIDSON

[![Build Status](https://travis-ci.org/safnuk/DIDSON.jl.svg?branch=master)](https://travis-ci.org/safnuk/DIDSON.jl)

[![Coverage Status](https://coveralls.io/repos/safnuk/DIDSON.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/safnuk/DIDSON.jl?branch=master)

[![codecov.io](http://codecov.io/github/safnuk/DIDSON.jl/coverage.svg?branch=master)](http://codecov.io/github/safnuk/DIDSON.jl?branch=master)

# Installation

First install the other unregistered packages it depends on, then the package itself:
```
julia> Pkg.clone("https://github.com/safnuk/BackgroundSegmenter.jl")
julia> Pkg.clone("https://github.com/safnuk/ObjectTracker.jl")
julia> Pkg.clone("https://github.com/safnuk/DIDSON.jl")
```

The package uses python numpy for loading arrays and scikit-video for loading avi files, so those should be installed on the system.

To view the foreground segmentation and object tracking in action, use:
```
julia> using DIDSON
julia> view_clip(samples[1])
```
There are 7 different sample clips in the repo (i.e. use `samples[n]` for `1 <= n <= 7`), or you can use your own avi file or numpy array file directly.
