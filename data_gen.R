library(ggplot2)

gen_time_series <- function(theta=0.1, epsilon=0.0, trend=0.01, start_point=0.0) {
  noise_sd = 0.1
  step_size = 1
  seq_len = 50
  x <- seq(1, seq_len + 1, step_size)
  y <- sin(x*theta + epsilon) + rnorm(length(x), mean =0.0, sd = noise_sd) + 
    trend*x + start_point
  #ff <- as.matrix(t(c(y, theta)))
  return(data.frame(x = x, y=y))
}

df = gen_time_series(theta = 0.5)
ggplot(data = df, mapping = aes(x = x, y = y)) + geom_line()

generate_data <- function(filename = "train", num_samples = 100, 
                          theta = seq(0.05, 0.5, 0.05)) {
  index = 0
  for(th in theta) {
    for(i in 1:num_samples) {
      index += 1
      df <- gen_time_series(th)
      #data <- as.matrix(t(c(df$y)))
      write.table(data, file = paste(filename, '_features.csv', sep = ""),
                  sep = ",", col.names = FALSE, append = TRUE, row.names = FALSE)
      write.table(data.frame(th, index), file = paste(filename, '_labels.csv', sep = ""), sep = ',',
                  col.names = FALSE, append = TRUE, row.names = FALSE)
    }
  }
}

generate_data('data/train', num_samples = 1000, seq(0.05, 0.5, 0.05))
generate_data('data/test', num_samples = 200, theta = seq(0.05,0.25, 0.02))
generate_data('data/val', num_samples = 200, theta = seq(0.05, 0.2, 0.01))