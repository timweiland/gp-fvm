# GP-FVM: Probabilistic Finite Volume Method based on Affine Gaussian Process inference

![Tsunami simulation using GP-FVM](./results/sendai.gif)

The code repository for our paper.

GP-FVM implements a fully probabilistic analogue of the popular Finite Volume Method (FVM).
Concretely, we provide implementations for FVM observations based on grid-structured observations.
These reduce multi-dimensional integrals to products of one-dimensional integrals, enabling efficient closed-form inference instead of costly numerical quadrature.
The resulting linear systems can be solved either directly through a Cholesky decomposition or iteratively.
We provide various utilities for iterative solvers based on IterGP [3].
We build directly on the framework implemented in LinPDE-GP [1] and use utilities from ProbNum [2].

## Cloning

This repository includes git submodules. Therefore, please clone it via

```setup
git clone --recurse-submodules https://github.com/timweiland/gp-fvm
```

If you forgot the `--recurse-submodules` flag, simply run

```setup
git submodule update --init --recursive
```

## Requirements

Start with a clean [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) [environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using **Python 3.11**, e.g. through

```setup
conda create -n gp-fvm python=3.11
```

Activate the environment, then run

```setup
pip install -r dev-requirements.txt
```

You should be all set! If something does not work, please reach out to us.

## Running the experiments

Once you've cloned the repository and installed the requirements, you're ready to run the experiments. Move into the `experiments` directory. Then:

```setup
cp .env.example .env
```

and replace the values in `.env`.

You may download the PDEBench data by running `make`.

Afterwards, the subdirectories contain runscripts.
Adapt the parameters to your needs and run them.
For example, to run GP-FVM on an 1D Advection problem class, you may execute

```setup
python3 experiments/advection/run_1d_advection_fv.py --hdf $HDF --beta $BETA --N-pde-t $N_PDE_T --N-pde-x $N_PDE_X --results-path $OUTPUTFOLDER
```

For your convenience, there are also example Slurm job scripts that you can adapt to your needs.

## Results

![Benchmark of collocation vs. GP-FVM vs. deep learning](./results/benchmark_figure.png)

Red: GP-FVM; blue: collocation.
This figure was produced with the notebook `experiments/0004_pdebench_figure.ipynb`.
Note that you first need to actually run the corresponding benchmarks in `experiments/{advection, darcy, wave}`.

## Background: Affine GP inference

Solving a partial differential equation (PDE) can be framed as a machine learning task through the language of Gaussian processes.
We start with a GP **prior** over the solution of the PDE, which encodes our prior knowledge.
Then, we encode all of the constraints over the solution - i.e. the initial and boundary conditions as well as the PDE - as **linear observations** of the sample paths of our GP.
GPs are nice in the sense that they allow us to directly condition on this linear information.
We can form the GP posterior - with closed-form equations for the mean and covariance function - for which the sample paths fulfill the provided linear observations.

## Related work

[1] [LinPDE-GP](https://github.com/marvinpfoertner/linpde-gp): We directly build on the framework implemented in LinPDE-GP. In fact, the core FVM code is implemented in our LinPDE-GP fork.

[2] [ProbNum](https://github.com/probabilistic-numerics/probnum): At a lower level, both LinPDE-GP and our work build on ProbNum, particularly on their `LinearOperator` code and their covariance function definition. Our ProbNum fork implements some further required functionality, including PyTorch support for matrix-vector products on the GPU. 

[3] [IterGP](https://github.com/JonathanWenger/itergp): While we do not directly depend on their code, we do depend on the method itself, which we reimplemented in our LinPDE-GP fork.

## Contributing

If you find any bugs or if you need help, feel free to open an issue.

Pull requests (PRs) are welcome.
We will review them when we have time.
Please use [black](https://github.com/psf/black) to format your code; we may set up some form of CI in the future.