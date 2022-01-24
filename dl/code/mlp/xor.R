## Solve the XOR problem

## let's build a sigmoidal activation
g <- function(x) { 1 / (1+ exp(-x)) }

## be careful about interpreting gprime!!!
gprime <- function(x) { y <- g(x); y * (1 - y) }

#curve(g, xlim=c(-3, 3))
#curve(gprime, xlim=c(-3, 3))
bias = -1

epsilon = 0.5
input=diag(8)
output=diag(8)
B=t(t(rep(bias,8)))
data=cbind(input,B,output)
data
inputs = t(data[,1:8])
targets = t(data[,10:17])


I=8 #excluding bias
J=2 #eclusing bias
K=8

W1 = matrix(runif(J*(I+1)), J, I+1) #+1 for bias
W2 = matrix(runif(K*(J+1)), K, J+1)



z_j = matrix(0, J+1, I)
delta_j = matrix(0,J,I)

nepoch = 100000
errors = rep(0, nepoch)

for (epoch in 1:nepoch) {
  DW1 = matrix(0, J, I)
  DW2 = matrix(0, K, J)

  epoch_err = 0.0


  for (i in 1:ncol(inputs)) {
    ## forward activation
    z_i = inputs[,i] # keep as col vector
    t_k = targets[,i]
    
    ## input to hidden
    x_j = W1 %*% z_i

    for (q in 1:J) {
      z_j[q,i] = g(x_j[q])
    }
    
    

    ## hidden to output

    x_k = W2 %*% z_j[,i]
    z_k = g(x_k)

    error = sum(0.5 * (t_k - z_k)^2)

    epoch_err = epoch_err + error
    
    ## backward error propagation.
    delta_k = gprime(x_k) * (t_k - z_k)
    DW2 = DW2 + outer(as.vector(delta_k), as.vector(z_j))

    ## Now get deltas for hidden layer.
    for (q in 1:J) {
      delta_j[q] = gprime(x_j[q]) * delta_k[1] * W2[1,q]
    }
    
    DW1 = DW1 + outer( delta_j, as.vector(z_i))
  }

  

  ## end of an epoch.
  errors[epoch] = epoch_err
  if (epoch %%1000==0) {
    print(epoch_err)
  }
  W2 = W2 + (epsilon*DW2)
  W1 = W1 + (epsilon*DW1)
  ##print(W1)
}
par(mar=c(1, 1, 1, 1))
plot(errors)




print_output <-  function() {

  z_j = matrix(0, J, 1)
  
  n_inputs = ncol(inputs)

  output = matrix(0, n_inputs, n_inputs)
  for (i in 1:n_inputs) {
    z_i = inputs[,i, drop=FALSE]
    x_j = W1 %*% z_i
    for (q in 1:J) {
      z_j[q] = g(x_j[q])
    }
   

    x_k = W2 %*% z_j
    z_k = g(x_k)
    ##browser()
    output[i,] = c(z_k)
  }
  #c(z_j[1:J]),
  return(output)
}
print_output()
t
