
import PIL.Image

try:
    from Tkinter import *
    import tkFileDialog as filedialog
except ImportError:
    from tkinter import *
    from tkinter import filedialog
import PIL.ImageTk
import cv2
import numpy as np
import imageio
import scipy
import pickle
import random
import os
import matplotlib.pyplot as plt
		
# Feature extractor
def extract_features(image_path, vector_size=32):
	image = imageio.imread(image_path, pilmode="RGB")
	try:
		# Using KAZE, cause SIFT, ORB and other was moved to additional module
		# which is adding addtional pain during install
		alg = cv2.KAZE_create()
		# Dinding image keypoints
		kps = alg.detect(image)
		# Getting first 32 of them. 
		# Number of keypoints is varies depend on image size and color pallet
		# Sorting them based on keypoint response value(bigger is better)
		kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
		# computing descriptors vector
		kps, dsc = alg.compute(image, kps)
		# Flatten all of them in one big vector - our feature vector
		dsc = dsc.flatten()
		# Making descriptor of same size
		# Descriptor vector size is 64
		needed_size = (vector_size * 64)
		if dsc.size < needed_size:
			# if we have less the 32 descriptors then just adding zeros at the
			# end of our feature vector
			dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
	except cv2.error as e:
		print('Error: ', e)
		return None
	return dsc
		
def batch_extractor(images_path, pickled_db_path="features.pck"):
	files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
	
	result = {}
	for f in files:
		print('Extracting features from image %s' % f)
		name = f.split('/')[-1].lower()
		result[name] = extract_features(f)
		
	# saving all our feature vectors in pickled file
	with open(pickled_db_path, 'wb') as fp:
		pickle.dump(result, fp)
		
		
class Matcher(object):
    def __init__(self, pickled_db_path="extract.pck"):
        with open(pickled_db_path, "rb") as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k in self.data:
            self.names.append(k)
            self.matrix.append(self.data[k])
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)
        
        '''def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)'''
        
    def cos_similarity(self, vector1, vector2):
        dotProduct = 0
        lengthV = 0
        lengthW = 0
        for i in range(len(vector1)):
            dotProduct += vector1[i]*vector2[i]
            lengthV += vector1[i]**2
            lengthW += vector2[i]**2
        cos = dotProduct / ((lengthV**0.5)*(lengthW**0.5))
        return cos

    def euclidian_length(self,vector1, vector2):
        d = 0
        for i in range(len(vector1)):
            d += (vector1[i] - vector2[i])**2
        return d**0.5

    def matchcos_similarity(self, image_path, topn=5):
        value = []
        dicts = {}
        sort = []
        vector = []
        features = extract_features(image_path)
        for i in self.matrix:
            value.append(self.cos_similarity(features, i))
            
        j = 0
        for i in value:
            dicts[i] = self.names[j]
            j += 1
        
        for k, v in sorted(dicts.items()):
            sort.append(v)
            vector.append(k)
        sort = sort[::-1]
        vector = vector[::-1]

        # getting top 5 records
        vector = vector[:topn]
        nearest_img = sort[:topn]
        print(nearest_img)
        return nearest_img,vector

    def matcheuclidean (self, image_path, topn = 5):
        value = []
        dicts = {}
        sort = []
        vector = []
        features = extract_features(image_path)
        for i in self.matrix:
            value.append(self.euclidian_length(features, i))
        j = 0
        for i in value:
            dicts[i] = self.names[j]
            j += 1

        for k, v in sorted(dicts.items()):
            sort.append(v)
            vector.append(k)
    
       # getting top 5 records
        nearest_img = sort[:topn]
        print(nearest_img)
        return nearest_img,vector

def show_img(path):
	img = imageio.imread(path, pilmode="RGB")
	plt.imshow(img)
	plt.figure()
		
def run(filename,met):
    images_path = 'reference'
    # batch_extractor(images_path)
    ma = Matcher('extract.pck')
    # print('Query image ==========================================')
    if(met == 2):
        t1=Label(text="Cosine Similarity").place(x=20,y=380)
        names,vector = ma.matchcos_similarity(filename, topn=5)
    elif(met == 1):
        t1=Label(text="Jarak Euclidean").place(x=20,y=380)
        names,vector = ma.matcheuclidean(filename, topn=5)
    # print('Result images ========================================')

    for i in names:
        # we got cosine distance, less cosine distance between vectors
        # more they similar, thus we subtruct it from 1 to get match value
        show_img(os.path.join(images_path, i))
    t1=Label(text='1. '+str(vector[0])).place(x=20,y=400)
    t1=Label(text='2. '+str(vector[1])).place(x=20,y=420)
    t1=Label(text='3. '+str(vector[2])).place(x=20,y=440)
    t1=Label(text='4. '+str(vector[3])).place(x=20,y=460)    
    t1=Label(text='5. '+str(vector[4])).place(x=20,y=480)
    plt.show()
    return vector


class App(Frame):
    def chg_image(self):
        self.img = PIL.ImageTk.PhotoImage(self.im)
        self.la.config(image=self.img, bg="#000000",width=300, height=300)
        
    def process(self):
        vector = run(self.filename,self.met)

    def insert(self,vector):
        t1.insert(END,vector[0])
        t2.insert(END,vector[1])
        t3.insert(END,vector[2])
    
    def met1(self):
        self.met = 1
    def met2(self):
        self.met = 2
    def open(self):
        self.filename = filedialog.askopenfilename()
        if self.filename != "":
            self.im = PIL.Image.open(self.filename)
        self.chg_image()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('Face Recognition')
        self.master.geometry('400x600')
        self.met= IntVar()
        self.labeltext = "Face Recognition by Data Scientist"
        fram = Frame(self)
        Button(fram, height=2,width=15, text='Process', command=self.process)
        Button(fram, text="Process", command=self.process).pack(side=LEFT)
        Button(fram, height=1,width=10, text='Open', command=self.open)
        Button(fram, text="Open File", command=self.open).pack(side=LEFT)
        Radiobutton(master, text='Euclidean distance', command = self.met1, value=1).place(x=20,y=350)
        Radiobutton(master, text='Cosine similarity', command=self.met2, value=2).place(x=220,y=350)
    
        fram.pack(side=TOP, fill=BOTH)

        self.la = Label(self)
        self.la.pack()

        self.pack()

if __name__ == "__main__":
    app = App(); app.mainloop()