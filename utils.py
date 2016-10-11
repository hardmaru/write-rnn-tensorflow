import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import random
import svgwrite
from IPython.display import SVG, display

def get_bounds(data, factor):
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0
    
  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)
    
  return (min_x, max_x, min_y, max_y)

# old version, where each path is entire stroke (smaller svg size, but have to keep same color)
def draw_strokes(data, factor=10, svg_filename = 'sample.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
    
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))

  lift_pen = 1
    
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
    
  command = "m"

  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "

  the_color = "black"
  stroke_width = 1

  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))

  dwg.save()
  display(SVG(dwg.tostring()))

def draw_strokes_eos_weighted(stroke, param, factor=10, svg_filename = 'sample_eos.svg'):
  c_data_eos = np.zeros((len(stroke), 3))
  for i in range(len(param)):
    c_data_eos[i, :] = (1-param[i][6][0])*225 # make color gray scale, darker = more likely to eos
  draw_strokes_custom_color(stroke, factor = factor, svg_filename = svg_filename, color_data = c_data_eos, stroke_width = 3)

def draw_strokes_random_color(stroke, factor=10, svg_filename = 'sample_random_color.svg', per_stroke_mode = True):
  c_data = np.array(np.random.rand(len(stroke), 3)*240, dtype=np.uint8)
  if per_stroke_mode:
    switch_color = False
    for i in range(len(stroke)):
      if switch_color == False and i > 0:
        c_data[i] = c_data[i-1]
      if stroke[i, 2] < 1: # same strike
        switch_color = False
      else:
        switch_color = True
  draw_strokes_custom_color(stroke, factor = factor, svg_filename = svg_filename, color_data = c_data, stroke_width = 2)

def draw_strokes_custom_color(data, factor=10, svg_filename = 'test.svg', color_data = None, stroke_width = 1):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
    
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))

  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y

  for i in range(len(data)):

    x = float(data[i,0])/factor
    y = float(data[i,1])/factor

    prev_x = abs_x
    prev_y = abs_y

    abs_x += x
    abs_y += y

    if (lift_pen == 1):
      p = "M "+str(abs_x)+","+str(abs_y)+" "
    else:
      p = "M +"+str(prev_x)+","+str(prev_y)+" L "+str(abs_x)+","+str(abs_y)+" "

    lift_pen = data[i, 2]

    the_color = "black"

    if (color_data is not None):
      the_color = "rgb("+str(int(color_data[i, 0]))+","+str(int(color_data[i, 1]))+","+str(int(color_data[i, 2]))+")"

    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill(the_color))
  dwg.save()
  display(SVG(dwg.tostring()))

def draw_strokes_pdf(data, param, factor=10, svg_filename = 'sample_pdf.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)

  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))

  abs_x = 25 - min_x 
  abs_y = 25 - min_y

  num_mixture = len(param[0][0])

  for i in range(len(data)):

    x = float(data[i,0])/factor
    y = float(data[i,1])/factor

    for k in range(num_mixture):
      pi = param[i][0][k]
      if pi > 0.01: # optimisation, ignore pi's less than 1% chance
        mu1 = param[i][1][k]
        mu2 = param[i][2][k]
        s1 = param[i][3][k]
        s2 = param[i][4][k]
        sigma = np.sqrt(s1*s2)
        dwg.add(dwg.circle(center=(abs_x+mu1*factor, abs_y+mu2*factor), r=int(sigma*factor)).fill('red', opacity=pi/(sigma*sigma*factor)))

    prev_x = abs_x
    prev_y = abs_y

    abs_x += x
    abs_y += y


  dwg.save()
  display(SVG(dwg.tostring()))



class DataLoader():
  def __init__(self, batch_size=50, seq_length=300, scale_factor = 10, limit = 500):
    self.data_dir = "./data"
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.scale_factor = scale_factor # divide data by this factor
    self.limit = limit # removes large noisy gaps in the data

    data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
    raw_data_dir = self.data_dir+"/lineStrokes"

    if not (os.path.exists(data_file)) :
        print("creating training data pkl file from raw source")
        self.preprocess(raw_data_dir, data_file)

    self.load_preprocessed(data_file)
    self.reset_batch_pointer()

  def preprocess(self, data_dir, data_file):
    # create data file from raw xml files from iam handwriting source.

    # build the list of xml files
    filelist = []
    # Set the directory you want to start from
    rootDir = data_dir
    for dirName, subdirList, fileList in os.walk(rootDir):
      #print('Found directory: %s' % dirName)
      for fname in fileList:
        #print('\t%s' % fname)
        filelist.append(dirName+"/"+fname)

    # function to read each individual xml file
    def getStrokes(filename):
      tree = ET.parse(filename)
      root = tree.getroot()

      result = []

      x_offset = 1e20
      y_offset = 1e20
      y_height = 0
      for i in range(1, 4):
        x_offset = min(x_offset, float(root[0][i].attrib['x']))
        y_offset = min(y_offset, float(root[0][i].attrib['y']))
        y_height = max(y_height, float(root[0][i].attrib['y']))
      y_height -= y_offset
      x_offset -= 100
      y_offset -= 100

      for stroke in root[1].findall('Stroke'):
        points = []
        for point in stroke.findall('Point'):
          points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset])
        result.append(points)

      return result

    # converts a list of arrays into a 2d numpy int16 array
    def convert_stroke_to_array(stroke):

      n_point = 0
      for i in range(len(stroke)):
        n_point += len(stroke[i])
      stroke_data = np.zeros((n_point, 3), dtype=np.int16)

      prev_x = 0
      prev_y = 0
      counter = 0

      for j in range(len(stroke)):
        for k in range(len(stroke[j])):
          stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
          stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
          prev_x = int(stroke[j][k][0])
          prev_y = int(stroke[j][k][1])
          stroke_data[counter, 2] = 0
          if (k == (len(stroke[j])-1)): # end of stroke
            stroke_data[counter, 2] = 1
          counter += 1
      return stroke_data

    # build stroke database of every xml file inside iam database
    strokes = []
    for i in range(len(filelist)):
      if (filelist[i][-3:] == 'xml'):
        print('processing '+filelist[i])
        strokes.append(convert_stroke_to_array(getStrokes(filelist[i])))

    f = open(data_file,"wb")
    pickle.dump(strokes, f, protocol=2)
    f.close()


  def load_preprocessed(self, data_file):
    f = open(data_file,"rb")
    self.raw_data = pickle.load(f)
    f.close()

    # goes thru the list, and only keeps the text entries that have more than seq_length points
    self.data = []
    counter = 0

    for data in self.raw_data:
      if len(data) > (self.seq_length+2):
        # removes large gaps from the data
        data = np.minimum(data, self.limit)
        data = np.maximum(data, -self.limit)
        data = np.array(data,dtype=np.float32)
        data[:,0:2] /= self.scale_factor
        self.data.append(data)
        counter += int(len(data)/((self.seq_length+2))) # number of equiv batches this datapoint is worth

    # minus 1, since we want the ydata to be a shifted version of x data
    self.num_batches = int(counter / self.batch_size)

  def next_batch(self):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    for i in range(self.batch_size):
      data = self.data[self.pointer]
      n_batch = int(len(data)/((self.seq_length+2))) # number of equiv batches this datapoint is worth
      idx = random.randint(0, len(data)-self.seq_length-2)
      x_batch.append(np.copy(data[idx:idx+self.seq_length]))
      y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))
      if random.random() < (1.0/float(n_batch)): # adjust sampling probability.
        #if this is a long datapoint, sample this data more with higher probability
        self.tick_batch_pointer()
    return x_batch, y_batch

  def tick_batch_pointer(self):
    self.pointer += 1
    if (self.pointer >= len(self.data)):
      self.pointer = 0
  def reset_batch_pointer(self):
    self.pointer = 0

