import visdom
import time
import numpy as np

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        self.index = {}
        self.log_text = ''
    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self
    def plot_many(self,d):
        for k, v in d.items():
            self.plot(k,v)
    def img_many(self, d):
        for k, v in d.items():
            self.img(k,v)
    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
