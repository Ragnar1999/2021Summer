f <- function(x) {
  x*x + (3/x) + 4
}

g <- function(x) {
  2*x - 3/(x*x)
}

curve(f)

error <- function(x, h) {
  est <- ( f(x+h) - f(x) ) / h
  abs(g(x) - est)
}

## Finite differences
hs = 10^-c(1:16)
errors = sapply(hs, function(h) error(0.1, h))
plot(hs, errors, xlab='step size',
     ylab='abs. error', log='xy')


######################################################################
## Dual numbers

dual.add = function(u, v) {
  c( u[1]+v[1],
     u[2]+v[2] )
}

dual.mult = function(u, v) {
  c(u[1]*v[1],
    u[2]*v[1] + u[1]*v[2])
}

dual.div = function(u, v) {
  c( u[1] / v[1],
    (u[2] / v[1]) - (u[1] * v[2]/ v[1]^2 ) )
}


dual.f = function(x) {
  a = dual.mult(x, x)
  b = dual.div( c(3,0) , x)
  c = dual.add(a, b)
  d = dual.add(c, c(4,0))
  d
}

dual.f(c(0.1,1))


x = c(0.1, 1)
y = dual.f(x)
f(x[1]) == y[1]


all.equal(y[1], f(x[1]))
all.equal(y[2], g(x[1]))
error = g(x[1]) - y[2]
options(digits=16)
print(error)
g(x[1]) == y[2]  ## Numerically demanding!

