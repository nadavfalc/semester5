#from multiprocessing.queues import JoinableQueue, Queue
from network import *
from preprocessor import *
import os
from my_queue import MyQueue
from multiprocessing import  JoinableQueue, Queue, cpu_count
import numpy as np

class IPNeuralNetwork(NeuralNetwork):
    
    def __init__(self, sizes=None, learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, matmul=np.matmul):
        # shared between workers
        self.result = Queue()
        super().__init__(sizes,learning_rate,mini_batch_size,number_of_batches,epochs,matmul)

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        #note: we will have the exact amount of data for all the workers at all times
        num_of_samples = self.epochs * self.number_of_batches * self.mini_batch_size
        # num of workers = num of cpu's
        numCPUs = int(os.environ['SLURM_CPUS_PER_TASK'])
        print("cpus: ", numCPUs)
        #spread equally images for workers
        samples_per_worker = num_of_samples // numCPUs

        # 1. Create Workers
		# (Call Worker() with self.mini_batch_size as the batch_size)
        #give each Worker all the training data and also and index. 
        #Thats the way we will decrease the synchronization between the threads        
        # 1. Create Workers
		# (Call Worker() with self.mini_batch_size as the batch_size)
        workers = [Worker(JoinableQueue(), self.result, training_data, self.mini_batch_size)
                     for _ in range(numCPUs)]
        for w in workers:
            w.start()
        # 2. Set jobs
        num_of_samples = self.epochs * self.number_of_batches * self.mini_batch_size
        #spread equally images for workers
        samples_per_worker = num_of_samples // numCPUs
        for w in workers:
            indexes = np.random.choice(len(training_data[0]), size=samples_per_worker)
            for i in indexes:
                w.jobs.put((training_data[0][i], training_data[1][i]))
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        for w in workers:
            w.jobs.put(None)

        super().fit(training_data, validation_data)

        for worker in workers:
            worker.join()
        
        '''
        count = 0
        # 3. Stop Workers
        for w in workers:
            print(count)
            w.terminate()
            count += 1
        '''
        
        
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches 
        here OR in Worker.run()
        '''
        data_new = []
        labels_new = []

        #gather right amount of data and labels and call super.create_batches
        for i in range(self.number_of_batches):
            for j in range(self.mini_batch_size):
                # get next modified data
                result_tuple = self.result.get()
                data_new.append(result_tuple[0])
                labels_new.append(result_tuple[1])
        return super().create_batches(np.array(data_new), np.array(labels_new), batch_size)


        

    
