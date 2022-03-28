########################################

# Data
softmax = c(221,236,287,416)
linear = c(1738,3004,5373,10071)
seq_len = c(8,16,32,64)
seq_len_quad = seq_len^2

########################################

# Approximate coefficients

softmaxlm = lm(softmax~seq_len_quad)
linearlm = lm(linear~seq_len)
b0_softmax = summary(softmaxlm)$coefficients[1]
b1_softmax = summary(softmaxlm)$coefficients[2]
b0_linear = summary(linearlm)$coefficients[1]
b1_linear = summary(linearlm)$coefficients[2]

# equation: b0_softmax+b1_softmax*x^2 = b0_linear+b1_linear*x

# Constructing Quadratic Formula
# https://rpubs.com/kikihatzistavrou/80124
result <- function(a,b,c){
  if(delta(a,b,c) > 0){ # first case D>0
    x_1 = (-b+sqrt(delta(a,b,c)))/(2*a)
    x_2 = (-b-sqrt(delta(a,b,c)))/(2*a)
    result = c(x_1,x_2)
  }
  else if(delta(a,b,c) == 0){ # second case D=0
    x = -b/(2*a)
  }
  else {"There are no real roots."} # third case D<0
}

# Constructing delta
delta<-function(a,b,c){
  b^2-4*a*c
}
# Constructing delta
delta<-function(a,b,c){
  b^2-4*a*c
}
#######################################


# Function-Call
x = result(b1_softmax,-b1_linear,b0_softmax - b0_linear)
#ceiling(x[1]) == 3149



#######################################


#Plot
xtick = c("4","6","8","10","12")
power_x = 3:13
seq_len_new = 2^(power_x)
runtime_softmax = b0_softmax + b1_softmax*seq_len_new^2
power_y_softmax = log(runtime_softmax,2)
runtime_linear = b0_linear + b1_linear*seq_len_new
power_y_linear = log(runtime_linear,2)

plot(power_x,power_y_softmax,type="b",col="red",pch=2, 
     ylim = c(5,25), cex=1, xlab = 'log(seq_len)',
     ylab = 'Runtime in log(second)', xaxt="n")
axis(1, at = c(4,6,8,10,12), labels = xtick)
par(new=TRUE)
plot(power_x,power_y_linear, type="b",
     col="black",pch=1, cex=1, axes =FALSE,
     xlab='', ylab='',ylim = c(5,25))

