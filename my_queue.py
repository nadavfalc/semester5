
class MyQueue(object):

    def __init__(self):
        self.queue = []

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.queue.append(msg)

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        self.queue.pop(0)

    