# GameInference.jl

Particle filtering techniques for online inference of other players in
intentions in general-sum differential games.

## Intro | Background

This package implements particle filtering techniques for online inference of
player intentions. Two sources of intention uncertainty are considered:

<dl>
<dt>Equilibrium uncertainty</dt>
<dd>
Uncertainty about the equilibrium strategies that other players will employ
to achieve a given objective.
</dd>
<dt>Objective uncertainty</dt>
<dd>
uncertainty about the objectives (cost functions in the differential game)
of other players.
</dd>
</dl>

Both sources of uncertainty are discussed in my [Master's thesis](). A thorough
discussion of issues of equilibrium uncertainty alone can be found in [our
paper](https://arxiv.org/abs/2002.04354):

```latex
@misc{peters2020inference,
    title={Inference-Based Strategy Alignment for General-Sum Differential Games},
    author={Lasse Peters and David Fridovich-Keil and Claire J. Tomlin and Zachary N. Sunberg},
    year={2020},
    eprint={2002.04354},
    archivePrefix={arXiv},
    primaryClass={eess.SY}
}
```

## Example: Inference-Based Strategy Alignment

The animation below shows an example of [inference-based strategy alignment](https://arxiv.org/abs/2002.04354).
This example shows the interaction of five players; a single robot (blue, starting on the left) and four humans.
The dynamics of each player are those of a 4D-unicycle and each player wishes to reach their goal on the other side of the intersection while avoiding collisions with others.
This problem can be cast as a general-sum differential game which can be solved to a non-cooperative equilibrium using [linear-quadratic approximations](https://arxiv.org/abs/1909.04694).
However, even for known costs for all players, there are multiple possible equilibrium solutions to this problem.
This repository provides a particle filtering techniques that admits to estimate the likelihood of different solutions.

Each particle corresponds to an equilibrium of the game and has a weight associated to it.
In the figure below, the true human strategies are shown with red dashed lines and the strategies that comprise the particle belief are shown as blue transparent lines.
The histogram below shows the distribution of weights in the particle belief.
As the robot interacts with human players and observes their decisions, it can infer the likelihood of the sampled equilibrium solutions.
After only a few time steps, the robot is able to recover the true human strategies and thus allows all players to efficiently reach their goals.

![](https://raw.githubusercontent.com/lassepe/GameInference.jl-results/81cdc0c5e72a813ac70299a40eba94dca795fb0a/uncertain-equilibria/gifs/5-player-closed-loop/5-player-planning-3.gif?token=ACM4E5VHSESXVPFES32A43K636ZHI)

```important
TODO

- add some gifs here
```

## Structure of this Repository

```important
TODO

- results have been moved to a seperate repository, to avoid pollution.
```


## Installation

In a Julia REPL from the project root:

```julia
using Pkg
pkg"activate ."
# adding iLQGames.jl manually, since it is currently not registered.
pkg"add https://github.com/lassepe/iLQGames.jl#v0.2.6"
pkg"instantiate"
```

In order to use multi-threading, make sure to [start Julia with multiple threads](https://docs.julialang.org/en/v1.6-dev/manual/multi-threading/#Starting-Julia-with-multiple-threads-1).
This setting can be verified from within the Julia `REPL` by looking at the output of `Threads.nthreads()`.

### Optional: Pre-Compilation into a System Image

Some of the methods in `iLQGames.jl` require some significant amount of
compilation for experiments with more players.
Though compile times have certainly improved since v1.4, it is advisable to
pre-compile some of the more compilation-heavy functions in to a system image
which can be loaded later to reduce startup times.
This step is useful if  you intend to run the experiments multiple times spread
out over multiple sessions and wish to try out different experiment parameters
(or modifications to the algorithm).

**Note:** If you only intend to try out this code for a single session, this
step is not necessary!

Pre-compilation is done as follows.

1. Install the
[PackageCompiler](https://github.com/JuliaLang/PackageCompiler.jl) if not
already present on your system:

```julia
]add PackageCompiler
```

2. Compile the system image. This will create the system image
   `precompile/ilqgames_dev.sysimg.so`.

```bash
julia precompile/compile_systemimage.jl
```

3. For all experiments, launch Julia as follows to load the custom system image.
```bash
julia --sysimage ./precompile/ilqgames_dev.sysimg.so
```

## Running Experiments

```important
TODO

- describe how to reproduce results
    1. create plots from raw data (CSV)
    2. reproduce the expirments
```
