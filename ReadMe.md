
Create conda env:
```shell
$ conda env create -f environment.yml
```

Remove conda env:
```shell
$ conda remove --name dyson --all -y
```


Create kernel for Jupyter:
```shell
$ conda activate dyson
(dyson)$ conda install ipykernel
(dyson)$ ipython kernel install --user --name=dyson_kernel
(dyson)$ conda deactivate
```