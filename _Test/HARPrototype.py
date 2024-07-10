from queue import Queue

class HARPrototype():
    def __init__(self):
        self.input_queue = Queue()#.put([tracks, meta_data])  
        self.output_queue = Queue()
        self._model = None

    def update(self):
        pass

    def check_event(self):
        pass

    def preprocess(self):
        pass

    def recognize(self):
        pass

    def shutdown(self):
        pass


# Input
#        recognize_thread = Thread(target=self.recognize, args=(self.input_queue, self.output_queue), daemon=False).start()

# Output
#        DB_thread = Thread(target=insert_event, args=(self.output_queue, conn, mq_conn),daemon=False).start() 
