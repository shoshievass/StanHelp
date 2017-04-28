### Helper Functions
logit <- function(x){
  return(log(x/(1-x)))
}

inv_logit <- function(x){
  return(1/(1+exp(-x)))
}

softmax <- function(x){
  exp(x)/sum(exp(x))
}

repmat = function(X,m,n){
  ##R equivalent of repmat (matlab)
  mx = dim(X)[1]
  nx = dim(X)[2]
  matrix(t(matrix(X,mx,nx*n)),mx*m,nx*n,byrow=T)
}
