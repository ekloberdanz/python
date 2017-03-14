class Stack:

    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        temp = self.stack[-1]
        self.stack = self.stack[0:-1]
        return temp

    def top(self):
        return self.stack[-1]

    def empty(self):
        if len(self.stack) == 0:
            return True
        else:
            return False


'''
stk = Stack()
stk.push(1)
stk.push(2)
stk.push(3)
while not stk.empty():
    print(stk.pop())

print(stk.empty())


'''