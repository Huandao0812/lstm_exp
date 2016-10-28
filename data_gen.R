library(ggplot2)
theta = 0.05
epsilon = 0.1
trend = 0.003
start_point = 0.0
noise_sd = 0.1
step_size = 0.5
seq_len = 100
x <- seq(1, seq_len + 1, step_size)
y <- sin(x*theta + epsilon) + rnorm(length(x), mean =0.0, sd = noise_sd) + trend*x + start_point
df <- data.frame(x = x, y=y)
ggplot(data = df, mapping = aes(x = x, y = y)) + geom_line()

ff <- as.matrix(t(y))
write.table(ff, file="data.csv", sep = ",", col.names = FALSE, append = TRUE)
gen_time_series <- function(theta=0.1, epsilon=0.0, trend=0.01, start_point=0.0) {
  noise_sd = 0.1
  step_size = 0.5
  seq_len = 100
  x <- seq(1, seq_len + 1, step_size)
  y <- sin(x*theta + epsilon) + rnorm(length(x), mean =0.0, sd = noise_sd) + 
    trend*x + start_point
  #ff <- as.matrix(t(c(y, theta)))
  return(data.frame(x = x, y=y))
}

generate_data <- function(filename = "data.csv", num_samples = 1000, theta = seq(0.05, 0.5, 0.05)) {
  for(t in theta) {
    for(i in 1:num_samples) {
      df <- gen_time_series(t)
      data = as.matrix(t(c(df$y, t)))
      write.table(data, file = filename, sep = ",", col.names = FALSE, append = TRUE)
    }
  }
}