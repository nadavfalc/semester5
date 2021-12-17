import multiprocessing 
from scipy import ndimage
from math import floor
import numpy as np

class Worker(multiprocessing.Process):
    def __init__(self, jobs, result, training_data=None, batch_size=0):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        image = ndimage.rotate(image, angle,reshape=False)
        return image

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        image = np.array([ndimage.interpolation.shift(row, -dx, cval = 0) for row in image])
        image = np.array([ndimage.interpolation.shift(col, -dy, cval = 0) for col in image.T]).T
        return image

    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        noise_array = np.array([int(np.random.uniform(-noise,noise)) for i in range(len(image)*len(image.T))])
        noise_matrix = noise_array.reshape(image.shape)
        image = image + noise_matrix
        my_function = lambda x: min(max(x,0),255)
        for i in range(len(image)):
            for j in range(len(image.T)):
                image[i][j] = my_function(image[i][j])
        return image

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        for row in range(len(image)):
            for col in range(len(image.T)):
                new_col_val = int(floor(row * tilt + col))
                if new_col_val >= len(image.T):
                    image[row][col] = 0
                else:
                    image[row][col] = image[row][new_col_val]
        return image

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        reshaped_image = np.reshape(image, (28,28))
        angle = np.random.random_integers(-8,8)
        dx,dy = np.random.random_integers(-2,2), np.random.random_integers(-2,2)
        noise = np.random.random_integers(-0.15,0.15)
        tilt = np.random.uniform(-0.1,0.1)

        #angle = 0
        #dx, dy = 0
        #noise = 0
        #tilt = 0
        modified_image = Worker.skew(Worker.add_noise(Worker.shift(Worker.rotate(reshaped_image,angle),dx,dy),noise),tilt)
        flat_data = modified_image.flatten()
        #flat_data = image
        return flat_data

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while 1:
            next_job = self.jobs.get()
            if next_job == None:
                break
            image, label = next_job
            modified_data = (self.process_image(image), label)
            self.result.put(modified_data)
            self.jobs.task_done()
'''/*
w1 = Worker([],[])
mat1 = np.array([[0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9],
                [0,1,2,3,4,5,6,7,8,9]])
print(Worker.rotate(mat1, 90))
print(Worker.shift(mat1,1,1))
print(Worker.add_noise(mat1,30))
print(Worker.skew(mat1,0.3))
'''