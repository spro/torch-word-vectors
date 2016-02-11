# trivial-classification

Uses a LookupTable to classify binary sentiment of trivial sentences (e.g. "i think it was good").

Network shape:

* One 2d vector per word with a lookup table embedding layer (`nn.LookupTable(n_words, 2)`)
* Sum embedded word vectors for single 2d vector per sentence (`nn.Sum()`)
* Sigmoid for output layer activation (`nn.Sigmoid()`)
* BCE criterion for sigmoid classification (`nn.BCECriterion()`)

The trained labels are 2d vectors of shape `<bad, good>` - the expected output for a "good" sentence is `<0, 1>` and for a "bad" sentence `<1, 0>`.

Example output (25 sentences, 1000 training iterations):

```
$ th classify.lua

is good ==> <0.0045, 0.9967>
is great ==> <0.0132, 0.9936>
so great ==> <0.0066, 0.9860>
is bad ==> <0.9926, 0.0045>
really bad ==> <0.9935, 0.0032>
just so ==> <0.2673, 0.4910>
```
