
class queue:

    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        temp = self.queue[0]
        self.queue = self.queue.pop()
        return temp

    def show(self):
        print(self.queue)


q = queue()

q.enqueue(1)
q.enqueue(2)
q.enqueue(3)

q.show()
print(q.dequeue())
'''

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

q = Queue
q.enqueue(2)
print(q)
'''