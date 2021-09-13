# CV1labs
Code for lab assignments for Computer Vision 1 (UvA)


## Lab 1

### Setting up

1. Clone the repository
2. Unzip images
```bash
cd lab1/photometric/
unzip photometric_images.zip
```
3. Managing dependencies: you can use `conda` or `virtualenv` - whatever floats your boat. I use `conda` (on Mac M1 ARM arch) and instructions for the same are as follows:
   ```bash
   conda create -n cv1labs
   conda activate cv1labs
   conda install python=3.9
   conda install numpy ipdb jupyterlab
   pip install opencv-python
   ```
