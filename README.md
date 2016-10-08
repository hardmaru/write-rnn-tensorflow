
## Generative Handwriting Demo using TensorFlow

![example](https://cdn.rawgit.com/hardmaru/write-rnn-tensorflow/master/svg/example.svg)

![example](https://cdn.rawgit.com/hardmaru/write-rnn-tensorflow/master/svg/many_examples.svg)

An attempt to implement the random handwriting generation portion of Alex Graves' [paper](http://arxiv.org/abs/1308.0850).

See my blog post at [blog.otoro.net](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow) for more information.

### How to use

I tested the implementation on TensorFlow r0.11 and Pyton 3.  I also used the following libraries to help:

```
svgwrite
IPython.display.SVG
IPython.display.display
xml.etree.ElementTree
argparse
pickle
```

### Training

You will need permission from [these wonderful people](http://www.iam.unibe.ch/fki/databases/iam-on-line-handwriting-database) people to get the IAM On-Line Handwriting data.  Unzip `lineStrokes-all.tar.gz` into the data subdirectory, so that you end up with `data/lineStrokes/a01`, `data/lineStrokes/a02`, etc.  Afterwards, running `python train.py` will start the training process.

A number of flags can be set for training if you wish to experiment with the parameters.  The default values are in `train.py`

```
--rnn_size RNN_SIZE             size of RNN hidden state
--num_layers NUM_LAYERS         number of layers in the RNN
--model MODEL                   rnn, gru, or lstm
--batch_size BATCH_SIZE         minibatch size
--seq_length SEQ_LENGTH         RNN sequence length
--num_epochs NUM_EPOCHS         number of epochs
--save_every SAVE_EVERY         save frequency
--grad_clip GRAD_CLIP           clip gradients at this value
--learning_rate LEARNING_RATE   learning rate
--decay_rate DECAY_RATE         decay rate for rmsprop
--num_mixture NUM_MIXTURE       number of gaussian mixtures
--data_scale DATA_SCALE         factor to scale raw data down by
--keep_prob KEEP_PROB           dropout keep probability
```

### Generating a Handwriting Sample

I've included a pretrained model in `/save` so it should work out of the box.  Running `python sample.py --filename example_name --sample_length 1000` will generate 4 .svg files for each example, with 1000 points.

### IPython interactive session.

If you wish to experiment with this code interactively, just run `%run -i sample.py` in an IPython console, and then the following code is an example on how to generate samples and show them inside IPython.

```
[strokes, params] = model.sample(sess, 800)
draw_strokes(strokes, factor=8, svg_filename = 'sample.normal.svg')
draw_strokes_random_color(strokes, factor=8, svg_filename = 'sample.color.svg')
draw_strokes_random_color(strokes, factor=8, per_stroke_mode = False, svg_filename = 'sample.multi_color.svg')
draw_strokes_eos_weighted(strokes, params, factor=8, svg_filename = 'sample.eos.svg')
draw_strokes_pdf(strokes, params, factor=8, svg_filename = 'sample.pdf.svg')

```

![example1a](https://cdn.rawgit.com/hardmaru/write-rnn-tensorflow/master/svg/example1.normal.svg)
![example1b](https://cdn.rawgit.com/hardmaru/write-rnn-tensorflow/master/svg/example1.color.svg)
![example1c](https://cdn.rawgit.com/hardmaru/write-rnn-tensorflow/master/svg/example1.multi_color.svg)
![example1d](https://cdn.rawgit.com/hardmaru/write-rnn-tensorflow/master/svg/example1.eos_pdf.svg)
![example1e](https://cdn.rawgit.com/hardmaru/write-rnn-tensorflow/master/svg/example1.pdf.svg)

Have fun-

## License

MIT


