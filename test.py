from tensorflow_probability import distributions as tfd

dist = tfd.Normal(loc=0, scale=0.25)
print(dist.prob(0) / 47.7)