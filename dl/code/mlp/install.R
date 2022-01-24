
## Step 0: check what your quota is on linux: "quota"

## 
## Step 1:
## install keras; you will be asked if you wish to install miniconda; say yes.

install.packages("keras")


## Step 2:
## This next line may take about ten minutes or so on subliminal.
## Could you report the time it takes.

system.time( keras::install_keras() )

## Step 3:
## Test it on MNIST.  This might take about five/ten minutes
## the first time (it will download MNIST into ~/.keras/datasets)

system.time ( rmarkdown::render('mnist_bp.Rmd') )



## Step 4: repeat step 0 to check you are not over quota.  
